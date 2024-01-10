import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 


###################
# LinearDeacy
###################

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate=0.0, name=None):
        super(LinearDecay, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.name = name 
        
    def __call__(self, step):
        with tf.name_scope(self.name or "LinearDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate,
                                                         name='initial_learning_rate')
            dtype = initial_learning_rate.dtype 
            
            end_learning_rate = tf.cast(self.end_learning_rate, dtype=dtype)
            
            global_steps_recomp = tf.cast(step, dtype=dtype)
            decay_steps_recomp = tf.cast(self.decay_steps, dtype=dtype)
            
            linear_decay = tf.divide(tf.subtract(initial_learning_rate, end_learning_rate), decay_steps_recomp)
            
            return tf.maximum(
                end_learning_rate,
                initial_learning_rate - tf.multiply(linear_decay, global_steps_recomp), 
                name = name
            )
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'initial_learning_rate': self.initial_learning_rate,
            'end_learning_rate': self.end_learning_rate,
            'decay_steps': self.decay_steps,
            'name': self.name
        })
        
        return config


####################
# ExponentialDecay
####################

class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None):
        super(ExponentialDecayScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ExponentialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.floor(p)
            return tf.multiply(
                initial_learning_rate, tf.pow(decay_rate, p), name=name
            )
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        })
        
        return config


##################
# CosineDecay
##################

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_target, warmup_steps=0, alpha=0.0, name=None):
        super(CosineDecayWithWarmupScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.alpha = alpha  # Controls the flatness of the cosine annealing
        self.name = name
        
    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / decay_steps
            tf_pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(decay_from_lr, decayed)
    
    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)
            
            if self.warmup_target is None:
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )
            
            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)
            
            global_step_recomp = tf.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )


    def get_config(self):
        config = super().get_config()
        
        config.update({
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'warmup_target': self.warmup_target,
            'alpha': self.alpha,
            'warmup_steps': self.warmup_steps,
            'name': self.name
        })
        
        return config


##########################
# PieceWiseConstantDecay
##########################

class PiecewiseConstantDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, boundaries, values, name):
        super(PiecewiseConstantDecayScheduler, self).__init__()
        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "PiecewiseConstant"):
            boundaries = tf.nest.map_structure(
                tf.convert_to_tensor, tf.nest.flatten(self.boundaries)
            )
            values = tf.nest.map_structure(
                tf.convert_to_tensor, tf.nest.flatten(self.values)
            )
            x_recomp = tf.convert_to_tensor(step)
            for i, b in enumerate(boundaries):
                if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
                    # We cast the boundaries to have the same type as the step
                    b = tf.cast(b, x_recomp.dtype.base_dtype)
                    boundaries[i] = b
            pred_fn_pairs = []
            pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
            pred_fn_pairs.append(
                (x_recomp > boundaries[-1], lambda: values[-1])
            )
            for low, high, v in zip(
                boundaries[:-1], boundaries[1:], values[1:-1]
            ):
                # Need to bind v here; can do this with lambda v=v: ...
                pred = (x_recomp > low) & (x_recomp <= high)
                pred_fn_pairs.append((pred, lambda v=v: v))

            # The default isn't needed here because our conditions are mutually
            # exclusive and exhaustive, but tf.case requires it.
            default = lambda: values[0]
            return tf.case(pred_fn_pairs, default, exclusive=True)
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'boundaries': self.boundaries,
            'values': self.values,
            'name': self.name
        })
        
        return config


#########################
# PolynomialDecay
#########################

class PolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False, name=None):
        super(PolynomialDecayScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "PolynomialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, 
                                                         name="initial_learning_rate")
            dtype = initial_learning_rate.dtype

            power = tf.cast(self.power, dtype=dtype)
            end_learning_rate = tf.cast(self.end_learning_rate, dtype=dtype)
            global_steps_recomp = tf.cast(step, dtype=dtype)
            decay_steps_recomp = tf.cast(self.decay_steps, dtype=dtype)

            if self.cycle:
                multiplier = tf.where(
                    tf.equal(global_steps_recomp, 0),
                    1.0,
                    tf.math.ceil(global_steps_recomp / self.decay_steps),
                )
                
                decay_steps_recomp = tf.multiply(decay_steps_recomp, multiplier)

            else:
                global_steps_recomp = tf.minimum(
                    global_steps_recomp, decay_steps_recomp
                )
            
            p = tf.divide(global_steps_recomp, decay_steps_recomp)
            polynomial_decay = tf.subtract(initial_learning_rate, end_learning_rate)
            polynomial_decay = tf.pow(tf.multiply(polynomial_decay, (1 - p)), power)

            
            return tf.add(polynomial_decay, end_learning_rate, name=name)
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps, 
            'end_learning_rate': self.end_learning_rate,
            'power': self.power,
            'cycle': self.cycle,
            'name': self.name
        })
        
        return config


################
# StepDecay
################


class StepDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_factor, decay_steps, name=None):
        super(StepDecayScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "StepDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate,
                                                         name="initial_learning_rate")
            dtype = initial_learning_rate.dtype 
            
            decay_factor = tf.cast(self.decay_factor, dtype=dtype)
            
            global_steps_recomp = tf.cast(step, dtype=dtype)
            decay_steps_recomp = tf.cast(self.decay_steps, dtype=dtype)
            
            d = tf.floor(global_steps_recomp, decay_steps_recomp)
            
            return tf.multiply(initial_learning_rate, tf.pow(decay_factor, d), name=name)
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'initial_learning_rate': self.initial_learning_rate, 
            'decay_factor': self.decay_factor,
            'decay_steps': self.decay_steps,
            'name': slef.name
        })
        
        return config