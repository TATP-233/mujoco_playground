#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def get_matching_files(directory: Path, suffix: str) -> set:
    """获取目录中所有匹配后缀的文件（绝对路径）"""
    if not directory.is_dir():
        raise ValueError(f"目录不存在: {directory}")
    return {
        f.resolve() for f in directory.iterdir()
        if f.is_file() and f.suffix == suffix
    }


def main():
    parser = argparse.ArgumentParser(description="监控目录中新增的特定后缀文件，并执行命令")
    parser.add_argument("directory", help="要监控的目录路径")
    parser.add_argument("suffix", help="文件后缀，例如 .txt 或 .log（注意带点）")
    parser.add_argument("interval_min", type=int, help="检查间隔（分钟）")
    parser.add_argument("command", help="发现新文件时要执行的命令（字符串）")
    parser.add_argument("--recursive", action="store_true", help="是否递归监控子目录（默认否）")

    args = parser.parse_args()

    dir_path = Path(args.directory).resolve()
    suffix = args.suffix if args.suffix.startswith('.') else '.' + args.suffix
    interval_sec = args.interval_min * 60
    command = args.command
    recursive = args.recursive

    # 初始扫描
    try:
        if recursive:
            def get_all_files(d):
                return {f.resolve() for f in d.rglob(f"*{suffix}") if f.is_file()}
            known_files = get_all_files(dir_path)
        else:
            known_files = get_matching_files(dir_path, suffix)
    except Exception as e:
        print(f"[错误] 初始化失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始扫描完成，共发现 {len(known_files)} 个 {suffix} 文件。")
    print(f"开始监控目录: {dir_path}（{'递归' if recursive else '非递归'}），每 {args.interval_min} 分钟检查一次...")

    while True:
        try:
            time.sleep(interval_sec)

            if recursive:
                current_files = {f.resolve() for f in dir_path.rglob(f"*{suffix}") if f.is_file()}
            else:
                current_files = get_matching_files(dir_path, suffix)

            new_files = current_files - known_files

            if new_files:
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 发现 {len(new_files)} 个新增文件:")
                for f in sorted(new_files):
                    print(f"  - {f}")
                print(f"正在执行命令: {command}")
                result = subprocess.run(command, shell=True)
                if result.returncode != 0:
                    print(f"[警告] 命令执行失败，返回码: {result.returncode}", file=sys.stderr)
                # 更新已知文件集，避免重复触发
                known_files = current_files
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 未发现新增文件。", end='\r')

        except KeyboardInterrupt:
            print("\n[INFO] 用户中断，退出监控。")
            break
        except Exception as e:
            print(f"[ERROR] 运行时出错: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
