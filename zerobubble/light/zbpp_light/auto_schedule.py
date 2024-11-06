from dataclasses import dataclass

@dataclass(eq=True)
class ScheduledNode:
    type: str
    stage: int
    minibatch: int

def auto_schedule(nstages, nmb):
    f = [0] * nstages
    b = [0] * nstages
    # W is a stack for each stage
    w = [[] for i in range(nstages)]
    result = [[] for i in range(nstages)]
    last_compute_id = [-1] * nstages
    def schedule_f(stage):
        if not stage == 0:
            result[stage].append(ScheduledNode(
                'RECV_FORWARD',
                stage=stage,
                minibatch=f[stage]))
        result[stage].append(
            ScheduledNode(
              type='F',
              stage=stage,
              minibatch=f[stage]))
        last_compute_id[stage] = len(result[stage]) - 1
        if not stage == nstages - 1:
            result[stage].append(ScheduledNode(
                'SEND_FORWARD',
                stage=stage,
                minibatch=f[stage]))
        f[stage] += 1
    def schedule_b(stage):
        if not stage == nstages - 1:
            result[stage].append(ScheduledNode(
                'RECV_BACKWARD',
                stage=stage,
                minibatch=b[stage]))
        result[stage].append(
            ScheduledNode(
              type='B',
              stage=stage,
              minibatch=b[stage]))
        last_compute_id[stage] = len(result[stage]) - 1
        if not stage == 0:
            result[stage].append(ScheduledNode(
                'SEND_BACKWARD',
                stage=stage,
                minibatch=b[stage]))
        w[stage].append(b[stage])
        b[stage] += 1
    def schedule_w(stage):
        assert last_compute_id[stage] != -1
        if result[stage][last_compute_id[stage]].type == 'B' and (stage == 0 or not f[stage] == nmb):
            result[stage][last_compute_id[stage]].type = 'BW'
        else:
            result[stage].append(
                ScheduledNode(
                  type='W',
                  stage=stage,
                  minibatch=w[stage][-1]))
            last_compute_id[stage] = len(result[stage]) - 1
        w[stage].pop(-1)
        
    
    for stage in range(nstages):
        num_warmup_microbatches = nstages - stage - 1
        num_warmup_microbatches = min(num_warmup_microbatches, nmb)
        remaining = nmb - num_warmup_microbatches
        for i in range(num_warmup_microbatches):
            schedule_f(stage)
        for i in range(remaining):
            schedule_f(stage)
            schedule_b(stage)
            if i >= stage:
                schedule_w(stage)
        for i in range(num_warmup_microbatches):
            schedule_b(stage)
            if remaining + i >= stage:
                schedule_w(stage)
        assert f[stage] == b[stage] == nmb
        while len(w[stage]) > 0:
            schedule_w(stage)

    for stage in range(nstages):
        print(' '.join([f'{x.type}{x.minibatch}' for x in result[stage] if x.type in {'F', 'B', 'W', 'BW'}]))
    return result


if __name__ == "__main__":
    auto_schedule(4, 12)
    auto_schedule(8, 4)
    