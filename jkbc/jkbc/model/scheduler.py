from functools import partial

import fastai
import fastai.callbacks as fcbs
import fastai.callback as fcb

import jkbc.types as t

# SCHEDULERS
NONE = 'none'
ONE_CYCLE = 'one_cycle'
ANNEALING_COS = 'annealing_cos'
ANNEALING_COS_WARMRESTART = 'annealing_cos_warmrestart'


def get_scheduler(config) -> t.Callable[[fastai.basic_train.Learner], None]:
    if NONE == config.scheduler:
        return lambda learn : None
    elif ONE_CYCLE == config.scheduler:
        return attach_one_cycle_scheduler(epochs=config.epochs, min_lr=config.learning_rate_min, max_lr=config.learning_rate)
    elif ANNEALING_COS == config.scheduler:
        return attach_annealing_cos_scheduler(epochs=config.epochs, min_lr=config.learning_rate_min, max_lr=config.learning_rate)
    elif ANNEALING_COS_WARMRESTART == config.scheduler:
        return attach_annealing_cos_warmrestart_scheduler(epochs=config.epochs, min_lr=config.learning_rate_min, max_lr=config.learning_rate,
                                             mom=config.momentum, n_cycles=cycle)
    else: raise NotImplementedError(f"{config.scheduler} scheduler not implemented")


# ## Schedulers

def attach_annealing_cos_scheduler(epochs: int,
                                   min_lr: float, max_lr: float) -> t.Callable[[fastai.basic_train.Learner], None]:
    def __scheduler(learner: fastai.basic_train.Learner):
        remove_all_schedulers_from_learner(learner)

        total_iterations = calc_total_iterations(learner, epochs)
        phase = fcbs.TrainingPhase(total_iterations).schedule_hp(
            'lr', (max_lr, min_lr), fcb.annealing_cos)
        scheduler = fcbs.GeneralScheduler(learner, [phase])

        add_scheduler_to_learner(learner, scheduler)
        
    return __scheduler


def attach_annealing_cos_warmrestart_scheduler(epochs: int, min_lr: float,
                                               max_lr: float, mom: float, n_cycles) -> t.Callable[[fastai.basic_train.Learner], None]:
    def __scheduler(learner: fastai.basic_train.Learner):
        remove_all_schedulers_from_learner(learner)

        total_iterations = calc_total_iterations(learner, epochs)
        splits = [2**n for n in range(n_cycles)]
        splits_norm = [s / sum(splits) for s in splits]
        phases = [(fcbs.TrainingPhase(total_iterations * splits_norm[i])
                   .schedule_hp('lr', (max_lr, min_lr), anneal=fcb.annealing_cos)
                   .schedule_hp('mom', mom)) for i in range(n_cycles)]
        scheduler = fcbs.GeneralScheduler(learner, phases)

        add_scheduler_to_learner(learner, scheduler)
        
    return __scheduler


def attach_one_cycle_scheduler(epochs: int, min_lr: float,
                               max_lr: float) -> t.Callable[[fastai.basic_train.Learner], None]:
    def __scheduler(learner: fastai.basic_train.Learner):
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

    return __scheduler


# Helpers

def calc_total_iterations(learner: fastai.basic_train.Learner, epochs: int) -> int:
    return len(learner.data.train_dl) * epochs


def add_scheduler_to_learner(learner: fastai.basic_train.Learner, scheduler: fcbs.GeneralScheduler) -> None:
    learner.callbacks.append(scheduler)


def remove_all_schedulers_from_learner(learner) -> None:
    learner.callbacks = list(filter(lambda cb: not isinstance(
        cb, fcbs.GeneralScheduler), learner.callbacks))
