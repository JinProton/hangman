# this file is used to search hyper-parameter in multiprocess

import subprocess
import time

def multi_run(commands, process_num):
    processes = [None] * process_num
    while len(commands) > 0:
        for idx in range(process_num):
            proc = processes[idx]
            if (proc is None) or (proc.poll() is not None):
                cmd = commands.pop(0)
                print(f">> run cmd: {cmd}")
                new_proc = subprocess.Popen(cmd, shell=True)
                processes[idx] = new_proc
                break
        
        time.sleep(0.1)
    for p in processes:
        if p is not None:
            p.wait()
    print("------------------------")


def run():
    w2s = [0.1, 0.2]
    w3s = [0.2, 0.3]
    w4s = [0.3, 0.4]
    w5s = [0.4, 0.5]
    w6s = [0.5, 0.6]
    w7s = [0.6, 0.7]
    qs = [0.65, 0.75, 0.85]

    commands = []
    idx = 0
    for w2 in w2s:
        for w3 in w3s:
            for w4 in w4s:
                for w5 in w5s:
                    for w6 in w6s:
                        for w7 in w7s:
                            for q in qs:
                                cmd = f"python game.py --idx {idx} --w2 {w2} --w3 {w3} --w4 {w4} --w5 {w5} --w6 {w6} --w7 {w7} --q {q}"
                                commands.append(cmd)
                                idx += 1
    multi_run(commands, 15)


if __name__ == "__main__":
    run()
