import matplotlib.pyplot as plt
import matplotlib.patches as patches

import copy
import multiprocessing
from dataclasses import dataclass
from typing import List, Set

import numpy as np

import torch
import pulp
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable
from pulp import constants as lp_const
from pulp import lpDot, lpSum


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

    # Manual ordering producing this kind of schedule:
    # fffffffbfbfbfbfbfbwbwbwbwbwbwbwwwwww
    #  fffffbfbfbfbfbfbfbfbwbwbwbwbwwwwwwww
    #   fffbfbfbfbfbfbfbfbfbfbwbwbwwwwwwwwww
    #    fbfbfbfbfbfbfbfbfbfbfbfbwwwwwwwwwwww
    # Returns the order index of each node on its own stage
    def manual_order(
        self, allow_bubble_before_first_b=False, prioritize_b=False, no_bubble_greedy=True
    ):
        order = [0] * self.nnodes
        f = [0] * self.nstages
        b = [0] * self.nstages
        w = [0] * self.nstages
        o = [0] * self.nstages
        m = [0] * self.nstages
        e = [0] * self.nstages
        t = [0] * self.nnodes
        max_mem = self.config.max_mem or [
          self.get_mem(self.get_id(0, stage, 0)) * self.nmb * 3 for stage in range(self.nstages)] 
        comm = self.config.cost_comm
        order_str = [""] * self.nstages
        stage_bubble = [0] * self.nstages

        def get_max_bubble():
            max_bubble = 0
            for bb in stage_bubble:
                max_bubble = max(max_bubble, bb)
            return max_bubble

        def put(stage_j, type_k):
            if type_k == 0:
                _i = f[stage_j]
            elif type_k == 1:
                _i = b[stage_j]
            else:
                _i = w[stage_j]
            _j = stage_j
            _id = self.get_id(type_k, _j, _i)
            _mem = self.get_mem(_id)
            _cost = self.get_cost(_id)
            assert m[_j] + _mem <= max_mem[stage_j]

            tmp = e[_j] + _cost
            no_bubble = tmp
            if _j > 0 and type_k == 0:
                tmp = max(tmp, t[self.get_id(0, _j - 1, _i)] + comm + _cost)
            if _j < self.nstages - 1 and type_k == 1:
                tmp = max(tmp, t[self.get_id(1, _j + 1, _i)] + comm + _cost)
            if f[_j] > 0:
                stage_bubble[_j] += tmp - no_bubble
            e[_j] = tmp
            t[_id] = tmp
            m[_j] += _mem
            order[_id] = o[_j]
            if type_k == 0:
                f[_j] += 1
            elif type_k == 1:
                b[_j] += 1
            else:
                w[_j] += 1
            o[_j] += 1
            fbw = "fbw"
            order_str[stage_j] += fbw[type_k]

        for i in range(self.nmb):
            if i == 0:
                for j in range(self.nstages):
                    put(j, 0)
                f_required = [0] * self.nstages
                last_t = 0
                for j in range(self.nstages - 1, -1, -1):
                    if j == self.nstages - 1:
                        last_t = t[self.get_id(0, j, i)] + self.get_cost(self.get_id(1, j, i))
                        continue
                    mem = m[j]
                    cost = e[j]
                    while True:
                        f_id = self.get_id(0, j, f[j] + f_required[j])
                        if f[j] + f_required[j] < self.nmb and mem + self.get_mem(f_id) <= max_mem[j]:
                            if allow_bubble_before_first_b:
                                if cost + self.get_cost(f_id) > last_t + comm:
                                    break
                            else:
                                if cost >= last_t + comm:
                                    break
                            mem += self.get_mem(f_id)
                            cost += self.get_cost(f_id)
                            f_required[j] += 1
                        else:
                            break
                    last_t = max(cost, last_t + comm) + self.get_cost(self.get_id(1, j, i))
                for j in range(self.nstages):
                    while j > 0 and f_required[j] > 0 and f_required[j] >= f_required[j - 1] and f[j] + f_required[j] < self.nmb:
                        f_required[j] -= 1
                for j in range(self.nstages):
                    for _ in range(f_required[j]):
                        put(j, 0)
                for j in range(self.nstages - 1, -1, -1):
                    put(j, 1)
                continue
            f_required = [0] * self.nstages
            for j in range(self.nstages):
                if f[j] >= self.nmb:
                    continue
                if j + 1 < self.nstages and f[j] >= f[j + 1] + 2 and prioritize_b:
                    next_plus_fw = (
                        e[j + 1]
                        + self.get_cost(self.get_id(0, j + 1, f[j + 1]))
                        + self.get_cost(self.get_id(1, j + 1, b[j + 1]))
                        + comm
                    )
                    if e[j] >= next_plus_fw:
                        continue
                    f_id = self.get_id(0, j, f[j])
                    f_mem = self.get_mem(f_id)
                    w_cost, w_cnt = 0, 0
                    mem_with_w = m[j] + f_mem
                    while mem_with_w > max_mem[j] and w[j] + w_cnt < b[j]:
                        w_id = self.get_id(2, j, w[j] + w_cnt)
                        w_cost += self.get_cost(w_id)
                        mem_with_w += self.get_mem(w_id)
                        w_cnt += 1
                    if e[j] + self.get_cost(f_id) + w_cost <= next_plus_fw:
                        f_required[j] = 1
                        continue

                    w_cost, w_cnt = 0, 0
                    # mem_with_w = m[j]
                    # while w[j] + w_cnt < b[j]:
                    #     w_id = self.get_id(2, j, w[j] + w_cnt)
                    #     w_cost += self.get_cost(w_id)
                    #     mem_with_w += self.get_mem(w_id)
                    #     w_cnt += 1
                    # if e[j] + w_cost >= next_plus_fw:
                    #     continue
                    if next_plus_fw - (e[j] + w_cost) <= get_max_bubble() - stage_bubble[j]:
                        # TODO: can sample here
                        continue
                f_required[j] = 1
            for j in range(self.nstages - 2, -1, -1):
                f_required[j] = min(f_required[j], f_required[j + 1])
            for j in range(self.nstages):
                if f_required[j] == 0:
                    continue
                f_id = self.get_id(0, j, f[j])
                mem = self.get_mem(f_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j > 0:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(0, j - 1, f[j])] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(0, j - 1, f[j])] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(0, j - 1, f[j])] + comm
                        if (
                            t[self.get_id(0, j - 1, f[j])] + comm - e[j]
                            <= get_max_bubble() - stage_bubble[j]
                        ):
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                put(j, 0)
            for j in range(self.nstages - 1, -1, -1):
                assert b[j] == i
                b_id = self.get_id(1, j, b[j])
                mem = self.get_mem(b_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j + 1 < self.nstages:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(1, j + 1, i)] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(1, j + 1, i)] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(1, j + 1, i)] + comm
                        if (
                            t[self.get_id(1, j + 1, i)] + comm - e[j]
                            <= get_max_bubble() - stage_bubble[j]
                        ):
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                if j == 0 and f[j] == self.nmb:
                    while w[j] < b[j]:
                        put(j, 2)
                put(j, 1)

        for i in range(self.nstages):
            while w[i] < self.nmb:
                put(i, 2)
            # print(f"{' ' * i}{order_str[i]}  -> {e[i]}")

        for i in range(self.nstages):
            for j in range(self.nmb):
                f_id = self.get_id(0, i, j)
                b_id = self.get_id(1, i, j)
                w_id = self.get_id(2, i, j)
                f_cost = self.get_cost(f_id)
                b_cost = self.get_cost(b_id)
                w_cost = self.get_cost(w_id)
                assert t[b_id] >= t[f_id] + b_cost
                assert t[w_id] >= t[b_id] + w_cost, f"{i}-{j}, {t[w_id]} >= {t[b_id]} + {w_cost}"
                if i > 0:
                    assert t[f_id] >= t[self.get_id(0, i - 1, j)] + comm + f_cost, f"{i}-{j}"
                if i < self.nstages - 1:
                    assert t[b_id] >= t[self.get_id(1, i + 1, j)] + comm + b_cost

        # print(order)
        best_time = 0
        for i in range(self.nstages):
            time_i = (
                t[self.get_id(2, i, self.nmb - 1)]
                - t[self.get_id(0, i, 0)]
                + self.get_cost(self.get_id(0, i, 0))
            )
            best_time = max(best_time, time_i)

        return order, t, best_time


def initial_solution(graph):
    best_time, order, complete_time = None, None, None
    for allow_bubble_before_first_b in [True, False]:
        for prioritize_b in [True, False]:
            for no_bubble_greedy in [True, False]:
                order_t, complete_time_t, best_time_t = graph.manual_order(
                    allow_bubble_before_first_b=allow_bubble_before_first_b,
                    prioritize_b=prioritize_b,
                    no_bubble_greedy=no_bubble_greedy,
                )
                if best_time is None or best_time_t < best_time:
                    best_time = best_time_t
                    order = order_t
                    complete_time = complete_time_t

    return best_time, order, complete_time


def build_ilp(graph, goal):
    def populate_ancestors_using_gpu(parents):
        with torch.no_grad():
            m = torch.zeros(graph.nnodes, graph.nnodes)
            for i in range(graph.nnodes):
                for j in parents[i]:
                    m[j][i] = 1
            # m = m.cuda()
            ones = torch.ones(graph.nnodes, graph.nnodes)#.cuda()
            while True:
                om = (m @ m) + m
                om = torch.minimum(om, ones)
                if torch.equal(om, m):
                    break
                m = om
            m = (m > 0).cpu()
        return m
    graph.precede = populate_ancestors_using_gpu(graph.parents)

    best_time, order, complete_time = initial_solution(graph)
    # exit(0)
    prob = LpProblem()
    # Dependency matrix
    P = {}
    for i in range(graph.nnodes):
        for j in range(0, i):
            if graph.requires_order(i, j):
                P[(i, j)] = LpVariable(f'P[{i},{j}]', cat=lp_const.LpBinary)
                P[(j, i)] = 1 - P[(i, j)]
    # End time of each node
    F = LpVariable.dicts('F', (range(graph.nnodes),), cat=lp_const.LpContinuous)

    inf = (
        (
            max(graph.config.cost_f)
            + max(graph.config.cost_b)
            + max(graph.config.cost_w)
            + graph.config.cost_comm * 3
        )
        * graph.nstages
        * graph.nmb
    )

    # P[i, j] + P[j, i] = 1
    for i in range(graph.nnodes):
        F[i].setInitialValue(complete_time[i])
        for j in range(0, i):
            if graph.requires_order(i, j):
                P[(i, j)].setInitialValue(order[i] < order[j])

    first_f = graph.get_id(0, 0, 0)
    prob += F[first_f] >= graph.get_cost(first_f)
    M = []
    for i in range(graph.nnodes):
        # prob += F[i] >= graph.get_cost(i)
        mem = []
        cost_sum = []
        for pre in range(graph.nnodes):
            if pre == i:
                continue
            # F[i] >= F[j] + cost[i] if there's an edge from j -> i
            if pre in graph.parents[i]:
                prob += F[i] >= graph.get_cost(i) + F[pre] + (
                    graph.config.cost_comm if graph.get_stage(pre) != graph.get_stage(i) else 0
                )

            if graph.get_stage(pre) == graph.get_stage(i):
                if graph.precede[pre][i]:
                    # prob += F[i] >= graph.get_cost(i) + F[pre]
                    mem.append(graph.get_mem(pre))
                    cost_sum.append(graph.get_cost(pre))
                elif graph.precede[i][pre]:
                    pass
                else:
                    # Unclear relationship
                    # F[i] >= F[j] + cost[i] if P[j, i]
                    # Translates to ILP:
                    # F[i] >= F[j] + cost[i] - P[i, j] * inf
                    prob += F[i] >= graph.get_cost(i) + F[pre] - P[(i, pre)] * inf
                    mem.append(graph.get_mem(pre) * P[(pre, i)])
                    cost_sum.append(graph.get_cost(pre) * P[(pre, i)])

        mem_i = lpSum(mem) + graph.get_mem(i)
        M.append(mem_i)
        if graph.config.max_mem is not None:
            prob += mem_i <= graph.config.max_mem[i]

    # Optimization targets
    res = LpVariable('Result')
    # Best completion time for each node
    for i in range(graph.nnodes):
        cost_sum = []
        for after in range(graph.nnodes):
            if after == i or graph.get_stage(after) != graph.get_stage(i):
                continue
            if graph.precede[after][i]:
                continue
            elif graph.precede[i][after]:
                cost_sum.append(graph.get_cost(after))
            else:
                cost_sum.append(graph.get_cost(after) * P[(i, after)])
        stage_id = graph.get_stage(i)
        prob += res >= F[i] + lpSum(cost_sum) - F[graph.get_id(0, stage_id, 0)] + graph.get_cost(graph.get_id(0, stage_id, 0))

    # Solve compute
    for i in range(graph.nstages):
        if goal == 'iter_time':
            prob += res >= F[graph.get_id(2, i, graph.nmb - 1)]
        else:
            prob += res >= F[graph.get_id(2, i, graph.nmb - 1)] - F[graph.get_id(0, i, 0)] + graph.get_cost(graph.get_id(0, i, 0))
    # maximum chain
    prob.setObjective(res)
    return (prob, P, F, M)


def solve_ilp(prob):
    def get_solver(verbose=True, warmStart=False):
        msg = verbose
        time_limit = 30
        assert "PULP_CBC_CMD" in pulp.listSolvers(
            onlyAvailable=True
        ), "Please install ILP solvers by 'sudo apt install coinor-cbc'"
        solver = pulp.PULP_CBC_CMD(
            mip=True,
            msg=msg,
            warmStart=warmStart,
            timeLimit=time_limit,
            gapAbs=0.5,
            threads=multiprocessing.cpu_count() // 2,
        )
        return solver

    solver = get_solver(warmStart=True)
    prob.solve(solver)



def get_exact_solution(goal, s, n, t_fs, t_bs, t_ws, c, plot_args=None):
    assert goal in ['iter_time', 'stage_time']
    graph_conf = GraphConfig([0]*s, [0]*s, [0]*s, None, t_fs, t_bs, t_ws, c, 1)
    graph = Graph.build_graph(s, n, graph_conf)
    prob, _, F, _ = build_ilp(graph, goal)
    end_times = [pulp.value(F[i]) for i in range(len(F))]
    solve_ilp(prob)
    end_times = [pulp.value(F[i]) for i in range(len(F))]
    if plot_args is not None:
        assert isinstance(plot_args, dict)
        ax = plot_args.get('ax', None)
        plot_ilp(graph, F, s, n, t_fs, t_bs, t_ws, ax)
        if 'output' in plot_args:
            plt.savefig(plot_args['output'])
    # Returns: (1) iteration time; (2) the optimized goal time (i.e., may be max_stage_time)
    return float(max(end_times)), float(pulp.value(prob.objective))


def plot_ilp(graph, F, num_stages, num_batches, t_fs, t_bs, t_ws, ax=None) -> None:
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
    if ax is None:
        plt.figure(figsize=(8.5, 2.5))
        ax = plt.subplot(111)
    maxt = 0
    color_map = {'F': '#FCCCB3', 'B': '#CBE4E4', 'W': '#FBE7A3'}
    for i in range(num_stages):
        for j in range(len(local_order[i])):
            batch = local_order[i][j]
            x, y = batch.start_time, num_stages - i - 1
            rect = patches.Rectangle((x, y), batch.completion_time - batch.start_time, 1, linewidth=1, edgecolor='black', facecolor=color_map[batch.type])
            ax.add_patch(rect)
            maxt = max(maxt, batch.completion_time)
    normal_stage_total = sum([(t_fs[i] + t_bs[i] + t_ws[i]) * num_batches for i in range(num_stages)])
    usage = normal_stage_total / (maxt * num_stages)
    rect = patches.Rectangle((0, 0), maxt, num_stages, linewidth=1, edgecolor='black', facecolor='#F2F2F2', zorder=0)
    ax.add_patch(rect)
    ax.set_title(f"S={num_stages}, B={num_batches}, Total Time = {maxt}, Bubble Rate = {(1 - usage) * 100:.2f}%")
    ax.set_xlim(0, maxt)
    ax.set_ylim(0, num_stages)
    ax.set_yticks(np.arange(num_stages) + 0.5, [f"{i}" for i in range(num_stages - 1, -1, -1)])

