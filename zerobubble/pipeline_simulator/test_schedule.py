import numpy as np
import matplotlib.pyplot as plt
import random
import unittest
from unittest import TestCase
from pipeline_simulator.ilp_solver import get_exact_solution
from pipeline_simulator.schedule import PipelineSimulator
from pipeline_simulator.batches import *
from pipeline_simulator.policy import *


class PipelineScheduleTest(TestCase):
    def test_optimality(self):
        for i in range(1):
            s = random.randint(3, 4)
            n = random.randint(2, 5) * s
            c = 0 #random.randint(0, 20)
            comm_delays = {(i, i + 1): c for i in range(s - 1)}
            t_f, t_b, t_w = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
            t_fs = np.array([t_f] * s) + np.random.normal(0, 2, s).astype(np.int32)
            t_bs = np.array([t_b] * s) + np.random.normal(0, 2, s).astype(np.int32)
            t_ws = np.array([t_w] * s) + np.random.normal(0, 2, s).astype(np.int32)
            goal = "iter_time"
            print("="*20)
            print(s, n, t_fs, t_bs, t_ws)
            min_iter, objective = get_exact_solution(
                goal, 
                s, n, t_fs.tolist(), t_bs.tolist(), t_ws.tolist(), c,
                {"output": "ilp.png"}
            )
            update_times(t_fs, t_bs, t_ws, [1]*s)
            policy = OurPolicy(s)
            simulator = PipelineSimulator(s, n, policy, [], comm_delays, True)
            simulator.simulate()
            simulator.plot()
            plt.savefig("our.png")
            stage_time, iter_time = simulator.get_max_stage_time(), simulator.get_iter_time()    
            print(f"ILP: [iter={min_iter}, obj[{goal}]={objective}]")
            print(f"Ours: [iter={iter_time}, max_stage={stage_time}]")

    def test_pipeline_adaption(self):
        return
        for test_id in range(1):
            num_stages = random.randint(2, 10)
            num_batches = random.randint(3, 10) * num_stages
            t_f, t_b, t_w = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
            t_fs = np.array([t_f] * num_stages)
            t_bs = np.array([t_b] * num_stages)
            t_ws = np.array([t_w] * num_stages)
            comm_delays = {(i, i + 1): 0 if random.random() < 0.8 else random.randint(0, 30) for i in range(num_stages - 1)}
            update_times(t_fs, t_bs, t_ws, [1]*num_stages)
            print(f"S={num_stages}, N={num_batches}, t={t_fs, t_bs, t_ws}, c={comm_delays}")

            policy_ref = OurPolicy(num_stages)
            simulator_ref = PipelineSimulator(num_stages, num_batches, policy_ref, [], comm_delays, True)
            simulator_ref.simulate()
            ref_iter_time = simulator_ref.get_iter_time()

            init_fwds = get_adapted_warmup_fwds(num_stages, t_fs, comm_delays)
            policy = DeltaiPolicy(num_stages, init_fwds)
            simulator = PipelineSimulator(num_stages, num_batches, policy, [], comm_delays, True)
            simulator.simulate()
            test_iter_time =  simulator.get_iter_time()            
            print(f"TestIter ({test_iter_time}), RefIter ({ref_iter_time})")

            assert test_iter_time == ref_iter_time, f"TestIter ({test_iter_time}) != RefIter ({ref_iter_time})"
            print(f"Test {test_id} passed!")


if __name__ == "__main__":
    unittest.main()

# plt.figure(figsize=(9, 2.5))
# simulator.plot()
# plt.title("")
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.bar(0, 0, label="Forward (F)", color=ForwardBatch(0).color, edgecolor='black')
# plt.bar(0, 0, label="Backward Input (B)", color=BackwardInputBatch(0).color, edgecolor='black')
# plt.bar(0, 0, label="Backward Weight (W)", color=BackwardWeightBatch(0).color, edgecolor='black')
# plt.xlabel("Time (ms)", fontdict={"fontsize": 16})
# plt.ylabel("Stages", fontdict={"fontsize": 16})
# legend = plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=15, ncols=3, frameon=False)
# plt.tight_layout(pad=1.0, rect=(0, 0, 1, 0.9)) 
# plt.savefig("schedule.pdf")

