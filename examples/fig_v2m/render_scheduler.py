#!/usr/bin/env python3
"""
Render Scheduler - 分配渲染任务到多 GPU 并行执行

Usage:
    python render_scheduler.py --config render_config.json --tasks task1,task2,task3
    python render_scheduler.py --config render_config.json --all  # 运行所有任务
    python render_scheduler.py --config render_config.json --tasks task1 --gpus 0,1,2,3  # 指定 GPU
"""

import argparse
import os
import sys
import subprocess
import time
import json
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass
from typing import List, Dict, Optional
import queue


@dataclass
class RenderTask:
    """渲染任务"""
    name: str
    config: Dict
    status: str = "pending"  # pending, running, completed, failed
    gpu_id: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_msg: Optional[str] = None


class GPUPool:
    """GPU 资源池管理"""
    
    def __init__(self, gpu_ids: List[int]):
        self.available_gpus = queue.Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put(gpu_id)
        self.lock = Lock()
    
    def acquire(self, timeout: float = None) -> Optional[int]:
        """获取一个可用的 GPU"""
        try:
            return self.available_gpus.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self, gpu_id: int):
        """释放 GPU"""
        self.available_gpus.put(gpu_id)
    
    def available_count(self) -> int:
        """返回可用 GPU 数量"""
        return self.available_gpus.qsize()


class RenderScheduler:
    """渲染调度器"""
    
    def __init__(
        self,
        config_path: str,
        blender_path: str = "blender",
        render_script: str = None,
        gpu_ids: List[int] = None,
        debug: bool = False,
        dry_run: bool = False,
        num_samples_override: int = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.blender_path = blender_path
        self.debug = debug
        self.dry_run = dry_run
        self.num_samples_override = num_samples_override
        
        # 渲染脚本路径
        if render_script is None:
            self.render_script = self.config_path.parent / "render_fbx_video3d.py"
        else:
            self.render_script = Path(render_script).resolve()
        
        # 加载配置
        with open(self.config_path, 'r') as f:
            self.full_config = json.load(f)
        
        # GPU 池
        if gpu_ids is None:
            gpu_ids = list(range(8))  # 默认 8 卡
        self.gpu_pool = GPUPool(gpu_ids)
        self.num_gpus = len(gpu_ids)
        
        # 任务列表
        self.tasks: Dict[str, RenderTask] = {}
        
        # 日志锁
        self.print_lock = Lock()
        
        # 中断处理
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        self.log("\n⚠️  收到中断信号，正在停止所有任务...")
        self.interrupted = True
        # 终止所有正在运行的进程
        for task in self.tasks.values():
            if task.process is not None and task.process.poll() is None:
                task.process.terminate()
    
    def log(self, msg: str, gpu_id: int = None):
        """线程安全的日志输出"""
        with self.print_lock:
            prefix = f"[GPU {gpu_id}]" if gpu_id is not None else "[Scheduler]"
            print(f"{prefix} {msg}")
    
    def get_task_names(self) -> List[str]:
        """获取所有可用的任务名称"""
        return list(self.full_config.get('tasks', {}).keys())
    
    def create_tasks(self, task_names: List[str]) -> List[RenderTask]:
        """创建任务列表"""
        tasks = self.full_config.get('tasks', {})
        defaults = self.full_config.get('defaults', {})
        
        result = []
        for name in task_names:
            if name not in tasks:
                self.log(f"⚠️  任务 '{name}' 不存在，跳过")
                continue
            
            # 合并默认配置和任务配置
            config = defaults.copy()
            config.update(tasks[name])
            
            task = RenderTask(name=name, config=config)
            self.tasks[name] = task
            result.append(task)
        
        return result
    
    def build_command(self, task: RenderTask, gpu_id: int) -> List[str]:
        """构建 Blender 渲染命令"""
        cmd = [
            self.blender_path,
            "-noaudio",
            "--background",
            "--python", str(self.render_script),
            "--",
            "--name", task.name,
            "--config", str(self.config_path),
        ]
        
        # 传递 num_samples (命令行覆盖 > 配置文件)
        if self.num_samples_override is not None:
            cmd.extend(["--num_samples", str(self.num_samples_override)])
        elif 'num_samples' in task.config:
            cmd.extend(["--num_samples", str(task.config['num_samples'])])
        
        if self.debug:
            cmd.append("--debug")
        else:
            cmd.append("--render")
        
        return cmd
    
    def run_task(self, task: RenderTask, gpu_id: int) -> bool:
        """运行单个渲染任务"""
        task.gpu_id = gpu_id
        task.status = "running"
        task.start_time = time.time()
        
        cmd = self.build_command(task, gpu_id)
        
        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        self.log(f"▶️  开始任务: {task.name}", gpu_id)
        
        if self.dry_run:
            self.log(f"   命令: {' '.join(cmd)}", gpu_id)
            self.log(f"   CUDA_VISIBLE_DEVICES={gpu_id}", gpu_id)
            time.sleep(0.5)  # 模拟运行
            task.status = "completed"
            task.end_time = time.time()
            return True
        
        # 创建输出目录
        out_dir = task.config.get('out', f'results/{task.name}')
        os.makedirs(out_dir, exist_ok=True)
        
        # 日志文件
        log_file = Path(out_dir) / "render.log"
        
        try:
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                f.flush()
                
                task.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.config_path.parent.parent.parent),  # 项目根目录
                )
                
                # 等待完成
                return_code = task.process.wait()
                
                task.end_time = time.time()
                duration = task.end_time - task.start_time
                
                if return_code == 0:
                    task.status = "completed"
                    self.log(f"✅ 完成任务: {task.name} (耗时 {duration:.1f}s)", gpu_id)
                    return True
                else:
                    task.status = "failed"
                    task.error_msg = f"Exit code: {return_code}"
                    self.log(f"❌ 任务失败: {task.name} (exit code: {return_code})", gpu_id)
                    return False
                    
        except Exception as e:
            task.status = "failed"
            task.error_msg = str(e)
            task.end_time = time.time()
            self.log(f"❌ 任务异常: {task.name} - {e}", gpu_id)
            return False
    
    def worker(self, task_queue: queue.Queue):
        """工作线程：从队列获取任务并执行"""
        while not self.interrupted:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if task is None:  # 终止信号
                break
            
            # 获取 GPU
            gpu_id = self.gpu_pool.acquire(timeout=1)
            while gpu_id is None and not self.interrupted:
                gpu_id = self.gpu_pool.acquire(timeout=1)
            
            if self.interrupted:
                task_queue.task_done()
                if gpu_id is not None:
                    self.gpu_pool.release(gpu_id)
                break
            
            try:
                self.run_task(task, gpu_id)
            finally:
                self.gpu_pool.release(gpu_id)
                task_queue.task_done()
    
    def run(self, task_names: List[str]) -> Dict[str, str]:
        """运行所有任务"""
        tasks = self.create_tasks(task_names)
        
        if not tasks:
            self.log("没有任务需要执行")
            return {}
        
        self.log(f"📋 共 {len(tasks)} 个任务，使用 {self.num_gpus} 个 GPU")
        
        # 创建任务队列
        task_queue = queue.Queue()
        for task in tasks:
            task_queue.put(task)
        
        # 添加终止信号
        for _ in range(self.num_gpus):
            task_queue.put(None)
        
        start_time = time.time()
        
        # 启动工作线程
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [executor.submit(self.worker, task_queue) for _ in range(self.num_gpus)]
            
            # 等待所有任务完成
            for future in futures:
                future.result()
        
        total_time = time.time() - start_time
        
        # 统计结果
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        
        self.log(f"\n{'='*50}")
        self.log(f"🏁 完成! 总耗时: {total_time:.1f}s")
        self.log(f"   ✅ 成功: {completed}")
        self.log(f"   ❌ 失败: {failed}")
        
        if failed > 0:
            self.log("\n失败的任务:")
            for task in self.tasks.values():
                if task.status == "failed":
                    self.log(f"   - {task.name}: {task.error_msg}")
        
        return {name: task.status for name, task in self.tasks.items()}


def main():
    parser = argparse.ArgumentParser(description="渲染任务调度器")
    parser.add_argument("--config", type=str, required=True, help="JSON 配置文件路径")
    parser.add_argument("--tasks", type=str, default=None, help="要运行的任务名称，逗号分隔")
    parser.add_argument("--all", action="store_true", help="运行所有任务")
    parser.add_argument("--list", action="store_true", help="列出所有可用任务")
    parser.add_argument("--gpus", type=str, default=None, help="使用的 GPU ID，逗号分隔 (默认: 0-7)")
    parser.add_argument("--blender", type=str, default="blender", help="Blender 可执行文件路径")
    parser.add_argument("--render-script", type=str, default=None, help="渲染脚本路径")
    parser.add_argument("--debug", action="store_true", help="调试模式 (低分辨率)")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际运行")
    parser.add_argument("--num-samples", type=int, default=None, help="覆盖所有任务的采样数")
    
    args = parser.parse_args()
    
    # 解析 GPU 列表
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(8))
    
    scheduler = RenderScheduler(
        config_path=args.config,
        blender_path=args.blender,
        render_script=args.render_script,
        gpu_ids=gpu_ids,
        debug=args.debug,
        dry_run=args.dry_run,
        num_samples_override=args.num_samples,
    )
    
    # 列出所有任务
    if args.list:
        print("可用任务:")
        for name in scheduler.get_task_names():
            print(f"  - {name}")
        return
    
    # 确定要运行的任务
    if args.all:
        task_names = scheduler.get_task_names()
    elif args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
    else:
        print("错误: 请指定 --tasks 或 --all")
        parser.print_help()
        sys.exit(1)
    
    # 运行
    results = scheduler.run(task_names)
    
    # 返回码
    if any(status == "failed" for status in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

