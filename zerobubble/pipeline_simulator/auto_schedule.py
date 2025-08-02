import torch
import pulp
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Set
from pipeline_simulator import batches
from pipeline_simulator.schedule import PipelineSimulator, calc_delta_x
from pipeline_simulator.policy import *


@dataclass
class GraphConfig:
    mem_f: List[float] = None
    mem_b: List[float] = None
    mem_w: List[float] = None
    max_mem: List[float] = None
    cost_f: List[int] = None
    cost_b: List[int] = None
    cost_w: List[int] = None
    cost_comm: int = 0
    print_scaling: int = 1

    def __post_init__(self):
        assert all([type(cost_f) is int for cost_f in self.cost_f])
        assert all([type(cost_b) is int for cost_b in self.cost_b])
        assert all([type(cost_w) is int for cost_w in self.cost_w])
        assert type(self.cost_comm) is int
        assert all([f + b + w == 0 for (f,b,w) in zip(self.mem_f, self.mem_b, self.mem_w)])


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    stage: int
    minibatch: int
    start_time: int
    completion_time: int
    rollback: bool = False


@dataclass
class Graph:
    nstages: int
    nmb: int
    nnodes: int
    config: GraphConfig
    parents: List[Set[int]] = None
    name: List[str] = None
    precede: torch.Tensor = None

    # ID mapping:
    # F[stage][minibatch]: 0..STAGE* MB
    # B[stage][minibatch]: STAGE* MB .. 2 * STAGE * MB
    # W[stage][minibatch]: 2 * STAGE* MB .. 3 * STAGE * MB

    def get_id(self, type, stage, mb):
        return type * (self.nstages * self.nmb) + stage * self.nmb + mb

    def get_stage(self, id):
        return (id // self.nmb) % self.nstages

    def get_cost(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.cost_f[stage], self.config.cost_b[stage], self.config.cost_w[stage]][type]

    def get_mem(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.mem_f[stage], self.config.mem_b[stage], self.config.mem_w[stage]][type]

    def requires_order(self, i, j):
        return (
            i != j
            and not self.precede[i][j]
            and not self.precede[j][i]
            and self.get_stage(i) == self.get_stage(j)
        )

    @classmethod
    def build_graph(cls, nstages, nmb, config):
        nnodes = nstages * nmb * 3
        g = Graph(nstages=nstages, nmb=nmb, nnodes=nnodes, config=config)
        parents = []
        name = []
        for type in range(3):
            for stage in range(nstages):
                for mb in range(nmb):
                    p = set()
                    if type == 0:
                        name.append(f'F{mb}')
                        if stage > 0:
                            p.add(g.get_id(type, stage - 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 1:
                        name.append(f'B{mb}')
                        if stage == nstages - 1:
                            p.add(g.get_id(0, stage, mb))
                        else:
                            p.add(g.get_id(type, stage + 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 2:
                        name.append(f'W{mb}')
                        p.add(g.get_id(1, stage, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    else:
                        assert False
                    parents.append(p)

        g.name = name
        g.parents = parents
        return g


def ilp_results(graph: Graph, F, comm_delay):
    typenames = ['F', 'B', 'W']
    local_order = []
    end_time = []
    for i in range(graph.nnodes):
        end_time.append(pulp.value(F[i]))
    for stage in range(graph.nstages):
        order = []
        for type in range(3):
            for mb in range(graph.nmb):
                id = graph.get_id(type, stage, mb)
                order.append(
                    ScheduledNode(
                        type=typenames[type],
                        stage=stage,
                        minibatch=mb,
                        start_time=end_time[id] - graph.get_cost(id),
                        completion_time=pulp.value(F[id]),
                    )
                )
        local_order.append(order)

    # For each F/B, append a send/recv node. The timestamp of recv node is the same as send node to guarrentee a global order.
    comm_id = {}
    comm_id_counter = 0
    for stage in range(graph.nstages):
        for node in local_order[stage]:
            if node.type == 'F' and node.stage != graph.nstages - 1:
                local_order[stage].append(
                    ScheduledNode(
                        type='SEND_FORWARD',
                        stage=stage,
                        minibatch=node.minibatch,
                        start_time=node.completion_time,
                        completion_time=node.completion_time,  # TODO: consider comm cost in completion time
                    )
                )
                local_order[stage + 1].append(
                    ScheduledNode(
                        type='RECV_FORWARD',
                        stage=stage + 1,
                        minibatch=node.minibatch,
                        start_time=node.completion_time,
                        completion_time=node.completion_time,  # TODO: consider comm cost in completion time
                    )
                )
                comm_id[local_order[stage][-1]] = comm_id_counter
                comm_id[local_order[stage + 1][-1]] = comm_id_counter
                comm_id_counter += 1
            if node.type == 'B' and node.stage != 0:
                local_order[stage].append(
                    ScheduledNode(
                        type='SEND_BACKWARD',
                        stage=stage,
                        minibatch=node.minibatch,
                        start_time=node.completion_time,
                        completion_time=node.completion_time,  # TODO: consider comm cost in completion time
                    )
                )
                local_order[stage - 1].append(
                    ScheduledNode(
                        type='RECV_BACKWARD',
                        stage=stage - 1,
                        minibatch=node.minibatch,
                        start_time=node.completion_time,
                        completion_time=node.completion_time,  # TODO: consider comm cost in completion time
                    )
                )
                comm_id[local_order[stage][-1]] = comm_id_counter
                comm_id[local_order[stage - 1][-1]] = comm_id_counter
                comm_id_counter += 1
    
    for stage in range(graph.nstages):
        # For nodes with the same timestamp on the same stage, communication will be prioritized.
        def even_breaker(x: ScheduledNode):
            # Compute nodes are always delayed.
            if x.type in ['F', 'B', 'W']:
                return comm_id_counter
            # For comm nodes, order by their unique comm id
            return comm_id[x]
        local_order[stage] = list(sorted(
            local_order[stage], key=lambda x: (x.start_time, even_breaker(x))
        ))
        # If a recv with intersects with previous computation, reorder them so that recv
        # is executed before computation and hence can be overlapped.
        for i in range(len(local_order[stage])):
            if i > 0 and local_order[stage][i - 1].type in {'F', 'B', 'W'} and \
                local_order[stage][i].type.startswith('RECV') and \
                "POST_VALIDATION" not in local_order[stage][i].type and \
                local_order[stage][i].start_time <= local_order[stage][i - 1].completion_time:
                (local_order[stage][i], local_order[stage][i - 1]) = (local_order[stage][i - 1], local_order[stage][i])
    return local_order


def search_best_simdelay(nstages: int, nmb: int, config: GraphConfig, delay_links: List, delay_time: float, step_size: int = 5):
    real_delay = {(i, i + 1): config.cost_comm for i in range(nstages - 1)}
    slow_stages = []
    for link in delay_links:
        real_delay[link] = delay_time
    path = os.getenv("OUT_DIR")
    path = path if path is not None else '.'
    best_sim_delay, best_iter_time = None, float("inf")
    for sim_delay in range(0, int(delay_time + step_size), int(step_size)):
        policy = OurPolicy(nstages)
        delay = {k: sim_delay for k in real_delay.keys() if real_delay[k] != 0}
        delay_simulator = PipelineSimulator(nstages, nmb, policy, slow_stages, delay, True)
        delay_simulator.simulate()
        schedule = delay_simulator.export()
        simu_4_plot = PipelineSimulator(nstages, nmb, FixedPolicy(nstages, None, schedule), slow_stages, real_delay, True)
        t = simu_4_plot.simulate()
        # simu_4_plot.export(f"{path}/simu_4_plot.txt")
        # print(f"[Search Schedule] sim_delay={delay}, iter_time={t - 1} delta_i={calc_delta_x(f'{path}/simu_4_plot.txt')}")
        if t < best_iter_time:
            best_iter_time = t
            best_sim_delay = sim_delay
    # print(f"[Search Schedule] best_sim_delay={best_sim_delay}")
    return best_sim_delay


def auto_schedule(nstages: int, nmb: int, config: GraphConfig, delay_links: List, delay_time: List[float]):
    if torch.distributed.get_rank() == 0:
        print(config.cost_f, config.cost_b, config.cost_w)
        print(f"{nstages} stages, {nmb} micro-batches")
    graph = Graph.build_graph(nstages, nmb, config)
    batches.update_times(config.cost_f, config.cost_b, config.cost_w)

    comm_delay = {(i, i + 1): config.cost_comm for i in range(nstages - 1)}
    slow_stages = []
    assert len(delay_links) == len(delay_time)
    for link, d in zip(delay_links, delay_time):
        comm_delay[link] = d
    if torch.distributed.get_rank() == 0:
        print(f"comm_delay: {comm_delay}")
    init_fwds = get_adapted_warmup_fwds(nstages, config.cost_f, comm_delay)
    policy = DeltaiPolicy(nstages, init_fwds)
    delay_simulator = PipelineSimulator(nstages, nmb, policy, slow_stages, comm_delay, True)
    delay_simulator.simulate()
    schedule = delay_simulator.export()
    simu_4_plot = PipelineSimulator(nstages, nmb, FixedPolicy(nstages, None, schedule), slow_stages, comm_delay, True)
    t = simu_4_plot.simulate()
    path = os.getenv("OUT_DIR")
    path = path if path is not None else '.'
    simu_4_plot.to_text(f"{path}/simu.txt")
    # print(f"[Simulation (ms)] {t - 1}")
    complete_time = delay_simulator.gen_schedule_graph_no_comm()
    return ilp_results(graph, complete_time, comm_delay)


if __name__ == "__main__":
    def simple_schedule(p,m,f,b,w,c,mem):
        return auto_schedule(p, m, GraphConfig(
            cost_f=[f]*p,
            cost_b=[b]*p,
            cost_w=[w]*p,
            cost_comm=c,
            mem_f=[2]*p,
            mem_b=[-1]*p,
            mem_w=[-1]*p,
            max_mem=[mem]*p,
            print_scaling=1000 if f > 1000 else 1
        ), [(1, 2)], 60)
    simple_schedule(4, 12, 10, 10, 10, 0, 0)
