from functools import partial

import fastai
import fastai.callbacks as fcbs
import fastai.callback as fcb

import jkbc.types as t


def get_named_schedulers(epochs: int, max_lr: float, moms: t.List[float],
                         cycles: t.List[int]) -> t.Dict[str, t.Callable[[fastai.basic_train.Learner], None]]:

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

def attach_annealing_cos_scheduler(learner: fastai.basic_train.Learner, epochs: int,
                                   min_lr: float, max_lr: float) -> None:

    remove_all_schedulers_from_learner(learner)

    total_iterations = calc_total_iterations(learner, epochs)
    phase = fcbs.TrainingPhase(total_iterations).schedule_hp(
        'lr', (max_lr, min_lr), fcb.annealing_cos)
    scheduler = fcbs.GeneralScheduler(learner, [phase])

    add_scheduler_to_learner(learner, scheduler)


def attach_annealing_cos_warmrestart_scheduler(learner: fastai.basic_train.Learner, epochs: int, min_lr: float,
                                               max_lr: float, mom: float, n_cycles) -> None:

    remove_all_schedulers_from_learner(learner)

    total_iterations = calc_total_iterations(learner, epochs)
    splits = [2**n for n in range(n_cycles)]
    splits_norm = [s / sum(splits) for s in splits]
    phases = [(fcbs.TrainingPhase(total_iterations * splits_norm[i])
               .schedule_hp('lr', (max_lr, min_lr), anneal=fcb.annealing_cos)
               .schedule_hp('mom', mom)) for i in range(n_cycles)]
    scheduler = fcbs.GeneralScheduler(learner, phases)

    add_scheduler_to_learner(learner, scheduler)


def attach_one_cycle_scheduler(learner: fastai.basic_train.Learner, epochs: int, min_lr: float,
                               max_lr: float) -> None:

    remove_all_schedulers_from_learner(learner)

    total_iterations = calc_total_iterations(learner, epochs)
    split = 0.3
    iterations_up = total_iterations * split
    iterations_down = total_iterations * (1 - split)

    up_phase = fcbs.TrainingPhase(iterations_up).schedule_hp(
        'lr', (min_lr, max_lr), anneal=fcb.annealing_cos)
    down_phase = fcbs.TrainingPhase(iterations_down).schedule_hp(
        'lr', (max_lr, min_lr), anneal=fcb.annealing_cos)
    scheduler = fcbs.GeneralScheduler(learner, [up_phase, down_phase])

    add_scheduler_to_learner(learner, scheduler)


# Helpers

def calc_total_iterations(learner: fastai.basic_train.Learner, epochs: int) -> int:
    return len(learner.data.train_dl) * epochs


def add_scheduler_to_learner(learner: fastai.basic_train.Learner, scheduler: fcbs.GeneralScheduler,
                             remove_other_schedulers: bool = False) -> None:
    if remove_other_schedulers:
        remove_all_schedulers_from_learner(learner)

    learner.callbacks.append(scheduler)


def remove_all_schedulers_from_learner(learner) -> None:
    learner.callbacks = list(filter(lambda cb: not isinstance(
        cb, fcbs.GeneralScheduler), learner.callbacks))
