import numpy as np
import matplotlib.pyplot as plt
import random
import unittest
import time
from unittest import TestCase
from pipeline_simulator.ilp_solver import get_exact_solution
from pipeline_simulator.schedule import PipelineSimulator
from pipeline_simulator.batches import *
from pipeline_simulator.policy import *


class PipelineScheduleTest(TestCase):
    def test_optimality(self):
        output_file = open("./result.csv", "a+")
        output_file.write("s, n, base, ilp_t, ilp_st, ilp_cost, our_t, our_st, our_cost\n")
        for (s, n) in [(3, 6), (4, 12), (6, 18), (8, 32)]:
            for base in [10, 15, 20, 25, 30]:
                for repeat in range(5):
                    c = random.random() * 20
                    comm_delays = {(i, i + 1): np.ceil(c) for i in range(s - 1)}
                    t_f, t_b, t_w = np.random.randint(base, 2 * base, 3)
                    t_fs = np.array([t_f] * s) + np.random.normal(0, 2, s)
                    t_bs = np.array([t_b] * s) + np.random.normal(0, 2, s)
                    t_ws = np.array([t_w] * s) + np.random.normal(0, 2, s)
                    goal = "iter_time"
                    print("="*20)
                    print(s, n, t_fs, t_bs, t_ws, c)

                    ilp_start = time.time()
                    min_iter, objective = get_exact_solution(
                        goal, 
                        s, n, t_fs.tolist(), t_bs.tolist(), t_ws.tolist(), c,
                        {"output": "ilp.png"}
                    )
                    ilp_end = time.time()

                    our_start = time.time()
                    update_times(np.ceil(t_fs), np.ceil(t_bs), np.ceil(t_ws), [1]*s)
                    policy = OurPolicy(s)
                    simulator = PipelineSimulator(s, n, policy, [], comm_delays, True)
                    simulator.simulate()
                    stage_time, iter_time = simulator.get_max_stage_time(), simulator.get_iter_time()
                    our_end = time.time()

                    print(f"ILP: [iter={min_iter}, obj[{goal}]={objective}]")
                    print(f"Ours: [iter={iter_time}, max_stage={stage_time}]")
                    output_file.write(f"{s}, {n}, {base}, {min_iter}, {objective}, {ilp_end - ilp_start}, {float(iter_time)}, {float(stage_time)}, {our_end - our_start}\n")

    def test_pipeline_adaption(self):
        for test_id in range(100):
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
    np.random.seed(2025)
    random.seed(2025)
    unittest.main()
