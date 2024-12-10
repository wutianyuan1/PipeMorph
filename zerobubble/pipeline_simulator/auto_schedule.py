import torch
import pulp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set
from pipeline_simulator import batches
from pipeline_simulator.schedule import PipelineSimulator
from pipeline_simulator.policy import GpipePolicy, PipeDreamPolicy, ZeroBubblePolicy, FixedPolicy


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


def ilp_results(graph: Graph, F):
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
    post_validation_time = 0
    for i in range(graph.nstages - 1, -1, -1):
        warmup_f_count = -1
        first_b_end = end_time[graph.get_id(1, i, 0)]
        for j in range(graph.nmb):
            if end_time[graph.get_id(0, i, j)] < first_b_end:
                warmup_f_count += 1
        assert warmup_f_count >= 0
        pv_id = warmup_f_count
        _id = graph.get_id(0, i, pv_id)
        _cost = graph.get_cost(_id)
        post_validation_time = max(post_validation_time, end_time[_id] - _cost - graph.config.cost_comm)
        # post_validation_time = 0
        # print(i, pv_id, post_validation_time)
        for it in ["RECV_", "SEND_", ""]:
            if i == 0 and it == "SEND_":
                continue
            if i == graph.nstages - 1 and it == "RECV_":
                continue
            # stage_ = i - 1 if it == "RECV_" else i
            stage_ = i
            local_order[stage_].append(ScheduledNode(
                type=it + "POST_VALIDATION",
                stage=stage_,
                minibatch=0,
                start_time=post_validation_time,
                completion_time=post_validation_time,
            ))
            comm_id[local_order[stage_][-1]] = comm_id_counter
            comm_id_counter += 1
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
        # print([(x.type, x.start_time, x.completion_time) for x in local_order[stage]])

    local_order_with_rollback = [[] for _ in range(graph.nstages)]
    for rank in range(graph.nstages):
        rollback_comm = set()
        if rank > 0:
            for node in local_order[rank - 1]:
                if node.type == "POST_VALIDATION":
                    break
                if node.type == "SEND_FORWARD":
                    rollback_comm.add(node.minibatch)
        for node in local_order[rank]:
            if node.type == "RECV_FORWARD" and node.minibatch in rollback_comm:
                rollback = True
                rollback_comm.remove(node.minibatch)
            else:
                rollback = False
            local_order_with_rollback[rank].append(ScheduledNode(
                type=node.type,
                stage=node.stage,
                minibatch=node.minibatch,
                start_time=node.start_time,
                completion_time=node.completion_time,
                rollback=rollback,
            ))
        assert len(rollback_comm) == 0
    return local_order_with_rollback


def auto_schedule(nstages: int, nmb: int, config: GraphConfig):
    # ==== A10 ====, Hidden=4096, MicroBSZ=1
    # [Rank0]{'F': 28, 'B': 30, 'W': 22}
    # [Rank2]{'F': 36, 'B': 37, 'W': 27}
    # [Rank1]{'F': 36, 'B': 36, 'W': 27}
    # [Rank3]{'F': 35, 'B': 34, 'W': 26}
    # config.cost_f = [28, 36, 36, 35]
    # config.cost_b = [30, 36, 37, 34]
    # config.cost_w = [22, 27, 27, 26]

    # ==== H800 ====, Hidden=6144, MicroBSZ=2
    # [Rank0]{'F': 12, 'B': 13, 'W': 8}
    # [Rank2]{'F': 16, 'B': 17, 'W': 11}
    # [Rank1]{'F': 16, 'B': 17, 'W': 11}
    # [Rank3]{'F': 15, 'B': 14, 'W': 10}
    # config.cost_f = [12, 16, 16, 15]
    # config.cost_b = [13, 17, 17, 14]
    # config.cost_w = [8, 11, 11, 10]

    config.cost_comm = 0
    print(config)

    graph = Graph.build_graph(nstages, nmb, config)
    batches.update_times(config.cost_f, config.cost_b, config.cost_w)

    # policy = PipeDreamPolicy(nstages)
    policy = ZeroBubblePolicy(nstages)
    # policy = FixedPolicy(nstages)
    comm_delay = {
        (i, i + 1): config.cost_comm for i in range(nstages - 1)
    }
    comm_delay[(0, 1)] = 5
    simulator = PipelineSimulator(nstages, nmb, policy, [], comm_delay, True)
    t = simulator.simulate()
    print(f"[Simulation (ms)] {t - 1}")
    simulator.plot()
    plt.savefig(f"./delay_{'_'.join([str(i) for i in list(comm_delay.values())])}.png")
    complete_time = simulator.gen_schedule_graph_no_comm()
    return ilp_results(graph, complete_time)


if __name__ == "__main__":
    # auto_schedule(4, 12, GraphConfig(cost_f=5, cost_b=6, cost_w=4, cost_comm=0, max_mem=10))
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
        ))
    simple_schedule(4, 4, 5, 6, 4, 2, 10)
