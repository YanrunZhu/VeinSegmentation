import subprocess
import os
from pathlib import Path
import time
import sys
import signal
from collections import deque
import threading

# --- 基本配置 ---
DATASET_ID = "230"
DIMENSION = "2d"
# !! Folds to train !!
FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]

# --- GPU 配置 ---
# !! 修改此处：配置你当前实际可用的GPU ID !!
AVAILABLE_GPUS = ["0", "1", "2"]  # 现在可用的是 GPU 0, 1, 2
# 确保这里的GPU ID是字符串类型。

# !! 每个训练折需要使用的GPU数量 (对于单卡训练，设置为1) !!
GPUS_PER_FOLD = 1

# --- nnU-Net 命令配置 ---
NNUNET_TRAIN_COMMAND = "nnUNetv2_train"  # 如果命令不在系统PATH中，请使用完整路径
# 日志文件存放的基础目录
LOG_DIR_BASE_FOR_MANAGER_STATUS = Path(f"./nnunet_parallel_sGPU_manager_status_Dataset230_{time.strftime('%Y%m%d_%H%M%S')}")
# 检查子进程状态的轮询间隔（秒）
POLL_INTERVAL_SECONDS = 15

# --- 全局列表，用于信号处理时追踪活动的 Popen 对象 ---
_active_processes_for_signal = []
_lock_for_active_processes = threading.Lock()

def graceful_shutdown(signum, frame):
    signal_name = signal.Signals(signum).name
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 接收到信号 {signal_name}。正在启动优雅关闭程序...")
    with _lock_for_active_processes:
        processes_to_terminate = list(_active_processes_for_signal)

    if not processes_to_terminate:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 没有活动的子进程需要终止。")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在终止 {len(processes_to_terminate)} 个活动的子进程...")

    for p in processes_to_terminate:
        if p.poll() is None:
            print(f"  发送 SIGTERM 到进程 PID {p.pid}...")
            p.terminate()

    termination_timeout_seconds = 20
    start_wait_time = time.time()
    all_terminated_gracefully = True

    for p in processes_to_terminate:
        if p.poll() is None:
            try:
                remaining_time = termination_timeout_seconds - (time.time() - start_wait_time)
                if remaining_time <= 0:
                    if p.poll() is None: p.kill(); print(f"  进程 PID {p.pid} (SIGTERM 超时) -> SIGKILL。"); all_terminated_gracefully = False
                    continue
                p.wait(timeout=max(0.1, remaining_time))
                print(f"  进程 PID {p.pid} 在 SIGTERM 后终止。")
            except subprocess.TimeoutExpired:
                if p.poll() is None: print(f"  进程 PID {p.pid} 未在规定时间内响应 SIGTERM。发送 SIGKILL..."); p.kill(); all_terminated_gracefully = False
            except Exception as e:
                print(f"  终止进程 PID {p.pid} 时发生错误: {e}"); all_terminated_gracefully = False
                if p.poll() is None: p.kill()

    if all_terminated_gracefully and processes_to_terminate:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 所有活动子进程已优雅终止。")
    elif processes_to_terminate:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 部分子进程可能需要强制终止 (SIGKILL)。")
    sys.exit(128 + signum)

def main_manager():
    if not AVAILABLE_GPUS:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: AVAILABLE_GPUS 列表为空。请配置要使用的GPU。")
        sys.exit(1)
    # 对于单卡训练，GPUS_PER_FOLD 期望为 1
    if GPUS_PER_FOLD != 1: 
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 配置错误: 当前脚本设计为单卡训练模式 (GPUS_PER_FOLD=1)，但GPUS_PER_FOLD被设置为 {GPUS_PER_FOLD}。")
        sys.exit(1)
    if len(AVAILABLE_GPUS) < GPUS_PER_FOLD:
         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: 可用GPU数量 ({len(AVAILABLE_GPUS)}) 少于每个fold所需的GPU数量 (1)。")
         sys.exit(1)

    LOG_DIR_BASE_FOR_MANAGER_STATUS.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] nnU-Net 并行单GPU训练管理器 - Dataset230")
    print(f"  (提示: 如需通过调整plans.json提高效率，请在运行此脚本前手动修改相应的plans文件)")
    print(f"  数据集 ID: {DATASET_ID}, 维度: {DIMENSION}")
    print(f"  待训练 Folds: {FOLDS_TO_TRAIN}")
    print(f"  总可用 GPUs: {AVAILABLE_GPUS} (共 {len(AVAILABLE_GPUS)} 张)")
    print(f"  每个Fold将使用 1 张 GPU 进行训练。")
    max_concurrent_jobs = len(AVAILABLE_GPUS) // GPUS_PER_FOLD 
    print(f"  预计最多可并行运行的训练任务数: {max_concurrent_jobs}")
    print(f"  管理器状态目录 (可选): {LOG_DIR_BASE_FOR_MANAGER_STATUS}")
    print("-" * 70)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    folds_pending_queue = deque(FOLDS_TO_TRAIN)
    free_gpus = sorted(list(set(AVAILABLE_GPUS))) 
    active_running_jobs = [] 
    
    completed_fold_count = 0
    total_folds_to_run = len(FOLDS_TO_TRAIN)
    if total_folds_to_run == 0:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 没有配置需要训练的Folds。脚本将退出。")
        sys.exit(0)
        
    failed_folds_info = {} 

    try:
        while completed_fold_count < total_folds_to_run:
            for job_info in list(active_running_jobs):
                process = job_info['process']
                fold_id = job_info['fold']
                gpus_assigned = job_info['gpus_assigned'] 

                if process.poll() is not None: 
                    exit_code = process.returncode
                    status_message = "成功" if exit_code == 0 else f"失败 (退出码: {exit_code})"
                    task_type_msg = "单GPU训练任务"
                    gpu_info_msg = f"GPU {gpus_assigned[0]}" 
                    
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_type_msg}完成: Fold {fold_id} (在{gpu_info_msg}上)。状态: {status_message}.")
                    
                    if exit_code != 0:
                        failed_folds_info[fold_id] = exit_code
                        print(f"  警告: Fold {fold_id} ({gpu_info_msg}) 返回非零退出码。请检查 nnU-Net results 目录中的日志。")

                    with _lock_for_active_processes:
                        if process in _active_processes_for_signal:
                            _active_processes_for_signal.remove(process)
                    
                    free_gpus.extend(gpus_assigned)
                    free_gpus.sort() 
                    
                    active_running_jobs.remove(job_info) 
                    completed_fold_count += 1

            while len(free_gpus) >= GPUS_PER_FOLD and folds_pending_queue: 
                gpus_for_new_job = sorted(free_gpus)[:GPUS_PER_FOLD] 
                
                for gpu_id in gpus_for_new_job:
                    free_gpus.remove(gpu_id)
                
                fold_to_run = folds_pending_queue.popleft()
                
                cuda_visible_devices_str = ",".join(gpus_for_new_job) 

                cmd_list_to_run = [
                    NNUNET_TRAIN_COMMAND,
                    DATASET_ID,
                    DIMENSION,
                    str(fold_to_run),
                    "-tr",
                    "nnUNetTrainerDA5",
                    "--c"
                ]
                
                current_process_env = os.environ.copy()
                current_process_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_str
                current_process_env["NNUNET_DO_NOT_COMPILE"] = "1"
                # 清理 LD_LIBRARY_PATH 以避免 cuDNN 版本冲突
                # PyTorch 自带 cuDNN，不需要系统路径中的旧版本
                if "LD_LIBRARY_PATH" in current_process_env:
                    # 移除包含 cudnn 的路径，保留其他必要的库路径
                    paths = current_process_env["LD_LIBRARY_PATH"].split(":")
                    filtered_paths = [p for p in paths if p and "cudnn" not in p.lower() and "cuda" not in p.lower()]
                    if filtered_paths:
                        current_process_env["LD_LIBRARY_PATH"] = ":".join(filtered_paths)
                    else:
                        # 如果过滤后为空，直接删除这个环境变量
                        del current_process_env["LD_LIBRARY_PATH"]

                task_type_msg = "单GPU训练任务"
                gpu_info_msg = f"GPU {gpus_for_new_job[0]}"

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 准备启动{task_type_msg}: Fold {fold_to_run} on {gpu_info_msg}")
                print(f"  命令: {' '.join(cmd_list_to_run)}")
                print(f"  环境变量: CUDA_VISIBLE_DEVICES={cuda_visible_devices_str}, NNUNET_DO_NOT_COMPILE=1")
                
                new_process = subprocess.Popen(
                    cmd_list_to_run,
                    env=current_process_env
                )
                with _lock_for_active_processes:
                    _active_processes_for_signal.append(new_process)
                
                active_running_jobs.append({
                    'process': new_process,
                    'fold': fold_to_run,
                    'gpus_assigned': gpus_for_new_job
                })
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 已启动{task_type_msg}: Fold {fold_to_run} on {gpu_info_msg} (PID: {new_process.pid})")

            if completed_fold_count >= total_folds_to_run:
                if not folds_pending_queue and not active_running_jobs: 
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 所有 {total_folds_to_run} 个预定 folds 已处理完毕。")
                    break 
            
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 主循环捕获到 KeyboardInterrupt。优雅关闭应已由信号处理程序处理。")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 管理器主循环发生意外错误: {e}")
        temp_procs_on_error = []
        with _lock_for_active_processes: temp_procs_on_error = list(_active_processes_for_signal)
        for p_err in temp_procs_on_error:
            if p_err.poll() is None: p_err.terminate()
        time.sleep(5) 
        for p_err in temp_procs_on_error:
            if p_err.poll() is None: p_err.kill()
        print("备用清理尝试完成。")
    finally:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 训练管理器脚本结束。")
        if failed_folds_info:
            print(f"  警告: 以下 folds 未成功完成 (返回非零退出码):")
            for f_id, ec in failed_folds_info.items():
                print(f"    Fold {f_id}: 退出码 {ec}")
        elif completed_fold_count == total_folds_to_run and not failed_folds_info: 
             print(f"  所有已处理的 folds ({FOLDS_TO_TRAIN}) 均报告成功完成。")

        print(f"  总共完成的 Folds: {completed_fold_count}/{total_folds_to_run}")

if __name__ == "__main__":
    main_manager()

