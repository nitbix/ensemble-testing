#!/usr/bin/python
"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

def sgd(param,learning_rate,gparam,mask,updates,current_cost,previous_cost):
    return param - learning_rate * gparam * mask

def old_rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
          eta_plus=1.2,eta_minus=0.5,max_delta=50, min_delta=1e-6):
    previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
    delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
    previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
    zero = T.zeros_like(param)
    one = T.ones_like(param)
    change = previous_grad * gparam

    new_delta = T.clip(
            T.switch(
                T.gt(change,0.),
                delta*eta_plus,
                T.switch(
                    T.lt(change,0.),
                    delta*eta_minus,
                    delta
                )
            ),
            min_delta,
            max_delta
    )
    new_previous_grad = T.switch(
            T.gt(change,0.),
            gparam,
            T.switch(
                T.lt(change,0.),
                zero,
                gparam
            )
    )
    inc = T.switch(
            T.gt(change,0.),
            - T.sgn(gparam) * new_delta,
            T.switch(
                T.lt(change,0.),
                zero,
                - T.sgn(gparam) * new_delta
            )
    )

    updates.append((previous_grad,new_previous_grad))
    updates.append((delta,new_delta))
    updates.append((previous_inc,inc))
    return param + inc * mask


def rprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
          eta_plus=1.01,eta_minus=0.1,max_delta=5, min_delta=1e-3):
    previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
    delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
    previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
    zero = T.zeros_like(param)
    one = T.ones_like(param)
    change = previous_grad * gparam

    new_delta = T.clip(
            T.switch(
                T.eq(gparam,0.),
                delta,
                T.switch(
                    T.gt(change,0.),
                    delta*eta_plus,
                    T.switch(
                        T.lt(change,0.),
                        delta*eta_minus,
                        delta
                    )
                )
            ),
            min_delta,
            max_delta
    )
    new_previous_grad = T.switch(
            T.eq(mask * gparam,0.),
            previous_grad,
            T.switch(
                T.gt(change,0.),
                gparam,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    gparam
                )
            )
    )
    inc = T.switch(
            T.eq(mask * gparam,0.),
            zero,
            T.switch(
                T.gt(change,0.),
                - T.sgn(gparam) * new_delta,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    - T.sgn(gparam) * new_delta
                )
            )
    )

    updates.append((previous_grad,new_previous_grad))
    updates.append((delta,new_delta))
    updates.append((previous_inc,inc))
    return param + inc * mask

def irprop(param,learning_rate,gparam,mask,updates,current_cost,previous_cost,
          eta_plus=1.5,eta_minus=0.25,max_delta=500, min_delta=1e-8):
    previous_grad = sharedX(numpy.ones(param.shape.eval()),borrow=True)
    delta = sharedX(learning_rate * numpy.ones(param.shape.eval()),borrow=True)
    previous_inc = sharedX(numpy.zeros(param.shape.eval()),borrow=True)
    zero = T.zeros_like(param)
    one = T.ones_like(param)
    change = previous_grad * gparam

    new_delta = T.clip(
            T.switch(
                T.eq(mask * gparam,0.),
                delta,
                T.switch(
                    T.gt(change,0.),
                    delta*eta_plus,
                    T.switch(
                        T.lt(change,0.),
                        delta*eta_minus,
                        delta
                    )
                )
            ),
            min_delta,
            max_delta
    )
    new_previous_grad = T.switch(
            T.eq(mask * gparam,0.),
            previous_grad,
            T.switch(
                T.gt(change,0.),
                gparam,
                T.switch(
                    T.lt(change,0.),
                    zero,
                    gparam
                )
            )
    )
    inc = T.switch(
            T.eq(mask * gparam,0.),
            zero,
            T.switch(
                T.gt(change,0.),
                - T.sgn(gparam) * new_delta,
                T.switch(
                    T.lt(change,0.),
                    zero,
#                    - T.sgn(gparam) * new_delta
                    T.switch( 
                        T.gt(current_cost, previous_cost),
                        - T.sgn(gparam) * new_delta,
                        zero
                    )
                )
            )
    )

    updates.append((previous_grad,new_previous_grad))
    updates.append((delta,new_delta))
    updates.append((previous_inc,inc))
    return param + inc * mask