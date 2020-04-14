import fastai.callbacks as fcb
import fastai
from functools import partial
import jkbc.types as t


def get_named_schedulers(epochs, max_lr, moms: t.List[float],
                         cycles: t.List[int]) -> t.Dict[str, t.Callable[[fastai.Learner], None]]:

    named_schedulers = {"one_cycle": partial(
        attach_one_cycle_scheduler, epochs=epochs, min_lr=0, max_lr=max_lr)}
    named_schedulers["annealing_cos"] = partial(
        attach_annealing_cos_scheduler, epochs=epochs, min_lr=0, max_lr=max_lr)

    for mom in moms:
        for cycle in cycles:
            name = f"annealing_cos_warmrestart::mom={mom}_cycles={cycle}"
            named_schedulers[name] = partial(attach_annealing_cos_warmrestart_scheduler,
                                             epochs=epochs, min_lr=0, max_lr=max_lr,
                                             mom=mom, n_cycles=cycle)
    return named_schedulers


# Schedulers

def attach_annealing_cos_scheduler(learner: fastai.Learner, epochs: int,
                                   min_lr: float, max_lr: float) -> None:

    remove_all_schedulers_from_learner(learner)

    total_iterations = calc_total_iterations(learner, epochs)
    phase = fcb.TrainingPhase(total_iterations).schedule_hp(
        'lr', (max_lr, min_lr), fcb.annealing_cos)
    scheduler = fcb.GeneralScheduler(learner, [phase])

    add_scheduler_to_learner(learner, scheduler)


def attach_annealing_cos_warmrestart_scheduler(learner: fastai.Learner, epochs: int, min_lr: float,
                                               max_lr: float, mom: float, n_cycles) -> None:

    remove_all_schedulers_from_learner(learner)

    total_iterations = calc_total_iterations(learner, epochs)
    splits = [2**n for n in range(n_cycles)]
    splits_norm = [s / sum(splits) for s in splits]
    phases = [(fcb.TrainingPhase(total_iterations * splits_norm[i])
               .schedule_hp('lr', (max_lr, min_lr), anneal=fcb.annealing_cos)
               .schedule_hp('mom', mom)) for i in range(n_cycles)]
    scheduler = fcb.GeneralScheduler(learner, phases)

    add_scheduler_to_learner(learner, scheduler)


def attach_one_cycle_scheduler(learner: fastai.Learner, epochs: int, min_lr: float,
                               max_lr: float) -> None:

    remove_all_schedulers_from_learner(learner)

    total_iterations = calc_total_iterations(learner, epochs)
    split = 0.3
    iterations_up = total_iterations * split
    iterations_down = total_iterations * (1 - split)

    up_phase = fcb.TrainingPhase(iterations_up).schedule_hp(
        'lr', (min_lr, max_lr), anneal=fcb.annealing_cos)
    down_phase = fcb.TrainingPhase(iterations_down).schedule_hp(
        'lr', (max_lr, min_lr), anneal=fcb.annealing_cos)
    scheduler = fcb.GeneralScheduler(learner, [up_phase, down_phase])

    add_scheduler_to_learner(learner, scheduler)


# Helpers

def calc_total_iterations(learner: fastai.Learner, epochs: int) -> int:
    return len(learner.data.train_dl) * epochs


def add_scheduler_to_learner(learner, scheduler: fcb.fcb.GeneralScheduler,
                             remove_other_schedulers: bool = True) -> None:
    if remove_other_schedulers:
        remove_all_schedulers_from_learner(learner)
    schedulers_in_learner = [cb for cb in learner.callbacks if isinstance(
        cb, (fcb.GeneralScheduler, fcb.Scheduler))]
    if schedulers_in_learner:
        raise Exception("Multiple Schedulers",
                        f"Learner already contains the following schedulers: \
                          {[sched.cb_name for sched in schedulers_in_learner]}")
    learner.callbacks.append(scheduler)


def remove_all_schedulers_from_learner(learner):
    learner.callbacks = list(filter(lambda cb: not isinstance(
        cb, (fcb.GeneralScheduler, fcb.Scheduler)), learner.callbacks))
