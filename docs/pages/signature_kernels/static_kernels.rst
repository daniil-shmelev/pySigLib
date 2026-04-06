Static Kernels
========================

A common approach when using signature kernels is to first lift the underlying ambient
space to a new feature space by means of a feature map, and then consider the signature
kernel in this feature space. Practically, this can be achieved by modifying the signature
kernel PDE to use a static kernel on the ambient space. Recall that the (standard) signature
kernel :math:`k_{x,y}` is the solution to the Goursat PDE

.. math::

    \frac{\partial^2 k_{x,y}}{\partial s \partial t} = \langle \dot{x}_s, \dot{y}_t \rangle k_{x,y}, \quad k_{x,y}(u, \cdot) = k_{x,y}(\cdot, v) = 1,

where a first order finite difference approximation yields

.. math::

    \frac{\partial^2 k_{x,y}}{\partial s \partial t} = \left( \langle x_s, y_t \rangle - \langle x_{s-1}, y_t \rangle - \langle x_s, y_{t-1} \rangle + \langle x_{s-1}, y_{t-1} \rangle \right) k_{x,y}, \quad k_{x,y}(u, \cdot) = k_{x,y}(\cdot, v) = 1.

If instead one considers a static kernel :math:`\kappa` on the ambient space, the equation becomes

.. math::

    \frac{\partial^2 k_{x,y}}{\partial s \partial t} = \left( \kappa(x_s, y_t) - \kappa(x_{s-1}, y_t) - \kappa(x_s, y_{t-1}) + \kappa(x_{s-1}, y_{t-1}) \right) k_{x,y}, \quad k_{x,y}(u, \cdot) = k_{x,y}(\cdot, v) = 1.

``pysiglib`` functions which utilise signature kernels accept :math:`\kappa` as an optional
parameter (``static_kernel``). By default, the standard linear kernel will be used. ``pysiglib``
provides implementations of the :ref:`linear kernel <linear-kernel-anchor>`,
:ref:`scaled linear kernel <scaled-linear-kernel-anchor>`, :ref:`RBF kernel <rbf-kernel-anchor>`,
:ref:`polynomial kernel <polynomial-kernel-anchor>`,
:ref:`Matern-1/2 kernel <matern12-kernel-anchor>`,
:ref:`Matern-3/2 kernel <matern32-kernel-anchor>`,
:ref:`Matern-5/2 kernel <matern52-kernel-anchor>` and
:ref:`rational quadratic kernel <rq-kernel-anchor>`,
which are documented below. In addition, one may define :ref:`custom kernels <custom-kernels-anchor>`.

.. code-block:: python

    import torch
    import pysiglib

    X = torch.rand((32, 100, 5))
    Y = torch.rand((32, 100, 5))

    # Default behaviour - linear kernel
    ker = pysiglib.sig_kernel(X, Y, dyadic_order=1)

    # Explicitly passed linear kernel - same as default behaviour
    static_kernel = pysiglib.LinearKernel()
    ker = pysiglib.sig_kernel(X, Y, dyadic_order=1, static_kernel=static_kernel)

    # RBF kernel
    static_kernel = pysiglib.RBFKernel(0.5)
    ker = pysiglib.sig_kernel(X, Y, dyadic_order=1, static_kernel=static_kernel)


Standard Kernels
------------------

.. _linear-kernel-anchor:
.. autoclass:: pysiglib.LinearKernel
.. _scaled-linear-kernel-anchor:
.. autoclass:: pysiglib.ScaledLinearKernel
.. _rbf-kernel-anchor:
.. autoclass:: pysiglib.RBFKernel
.. _polynomial-kernel-anchor:
.. autoclass:: pysiglib.PolynomialKernel
.. _matern12-kernel-anchor:
.. autoclass:: pysiglib.Matern12Kernel
.. _matern32-kernel-anchor:
.. autoclass:: pysiglib.Matern32Kernel
.. _matern52-kernel-anchor:
.. autoclass:: pysiglib.Matern52Kernel
.. _rq-kernel-anchor:
.. autoclass:: pysiglib.RationalQuadraticKernel
.. _custom-kernels-anchor:

Custom Kernels
------------------

In addition to the provided kernels, one can use a custom kernel by defining a child
class of the abstract base class ``pysiglib.StaticKernel`` and using the methods of
``pysiglib.Context`` to save objects for re-use in backpropagation. For example,
an implementation of ``pysiglib.LinearKernel`` is given below. When writing custom
kernels, it is very important to make them as efficient as possible, as computation
of the static kernel makes up a significant proportion of the overall computational
cost of signature kernels.

.. code-block:: python

    from pysiglib import StaticKernel

    class LinearKernel(StaticKernel):

    def __call__(self, ctx, x, y):
        dx = torch.diff(x, dim=1)
        dy = torch.diff(y, dim=1)
        ctx.save_for_backward(dx, dy)
        return torch.bmm(dx, dy.permute(0, 2, 1))

    def grad_x(self, ctx, derivs):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1] + 1, dy.shape[1]), dtype=torch.float64, device=derivs.device)
        out[:, 0, :] = 0
        out[:, 1:, :] = derivs
        out[:, :-1, :] -= derivs
        return torch.bmm(out, dy)

    def grad_y(self, ctx, derivs):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1], dy.shape[1] + 1), dtype=torch.float64, device=derivs.device)
        out[:, :, 0] = 0
        out[:, :, 1:] = derivs
        out[:, :, :-1] -= derivs
        return torch.bmm(out.permute(0, 2, 1), dx)

.. autoclass:: pysiglib.Context
   :members:

.. autoclass:: pysiglib.StaticKernel
   :members:
   :special-members: __call__
