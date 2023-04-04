from typing import Tuple, cast

import torch
import torch.nn as nn

import torchfilter
from torchfilter import types

from legged_gym.filters.filter_helpers import wrap_to_pi

torch.random.manual_seed(0)


class LinearDynamicsModel(torchfilter.base.DynamicsModel):
    """Forward model for our linear system. Maps (initial_states, controls) pairs to
    (predicted_state, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim, 
                    control_dim, 
                    observation_dim,
                    A, B, q_val, device,
                    ang_dims=None,
                    trainable: bool = False):
        super().__init__(state_dim=state_dim)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.device = device
        self.ang_dims = ang_dims

        self.A = A 
        self.B = B 
        self.Q_tril = torch.eye(state_dim, device=self.device, requires_grad=False) * q_val

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1], device=self.device))

    def forward(
        self,
        *,
        initial_states: types.StatesTorch,
        controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Forward step for a discrete linear dynamical system.

        Args:
            initial_states (torch.Tensor): Initial states of our system.
            controls (dict or torch.Tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """

        # Controls should be tensor, not dictionary
        assert isinstance(controls, torch.Tensor)
        controls = cast(torch.Tensor, controls)
        # Check shapes
        N, state_dim = initial_states.shape
        N_alt, control_dim = controls.shape
        assert self.A.shape == (state_dim, state_dim)
        assert N == N_alt

        # Compute/return states and noise values
        predicted_states = (self.A[None, :, :] @ initial_states[:, :, None]).squeeze(-1) + (
            self.B[None, :, :] @ controls[:, :, None]
        ).squeeze(-1)

        # wrap angular dimensions if they exist
        # if self.ang_dims is not None:
        #     predicted_states[:, self.ang_dims] = wrap_to_pi(predicted_states[:, self.ang_dims])

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        return predicted_states, self.Q_tril[None, :, :].expand((N, state_dim, state_dim))


class LinearKalmanFilterMeasurementModel(torchfilter.base.KalmanFilterMeasurementModel):
    """Kalman filter measurement model for our linear system. Maps states to
    (observation, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim, 
                    observation_dim, 
                    C, r_val, device,
                    ang_dims=None,
                    trainable: bool = False):
        super().__init__(state_dim=state_dim, observation_dim=observation_dim)
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.device = device
        self.ang_dims = ang_dims

        self.C = C 
        self.C_pinv = torch.pinverse(C)
        self.R_tril = torch.eye(observation_dim, device=self.device, requires_grad=False) * r_val
        if self.ang_dims is not None:
            self.R_tril[self.ang_dims, self.ang_dims] *= 0.5 # adjust the measurement covariance for angular component

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1], device=self.device))

    def forward(
        self, *, states: types.StatesTorch
    ) -> Tuple[types.ObservationsNoDictTorch, types.ScaleTrilTorch]:
        """Observation model forward pass, over batch size `N`.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, state_dim)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing expected observations
            and cholesky decomposition of covariance.  Shape should be `(N, M)`.
        """
        # Check shape
        N = states.shape[0]
        assert states.shape == (N, self.state_dim)

        # Compute output
        observations = (self.C[None, :, :] @ states[:, :, None]).squeeze(-1)
        scale_tril = self.R_tril[None, :, :].expand((N, self.observation_dim, self.observation_dim))

        # wrap angular dimensions if they exist
        # if self.ang_dims is not None:
        #     observations[:, self.ang_dims] = wrap_to_pi(observations[:, self.ang_dims])

        # Add output bias if trainable
        if self.trainable:
            observations += self.output_bias

        # Compute/return predicted measurement and noise values
        return observations, scale_tril


class LinearVirtualSensorModel(torchfilter.base.VirtualSensorModel):
    """Virtual sensor model for our linear system. Maps raw sensor readings to predicted
    states and uncertainties.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim, 
                        C, r_val, device,
                        ang_dims=None, trainable: bool = False):
        super().__init__(state_dim=state_dim)
        self.state_dim = state_dim
        self.device = device
        self.ang_dims = ang_dims

        self.C = C
        self.C_pinv = torch.pinverse(C)
        self.R_tril = torch.eye(observation_dim, device=self.device, requires_grad=False) * r_val
        if self.ang_dims is not None:
            self.R_tril[self.ang_dims, self.ang_dims] *= 0.5 # adjust the measurement covariance for angular component

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1], device=self.device))

    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Predicts states and uncertainties from observation inputs.

        Uncertainties should be lower-triangular Cholesky decompositions of covariance
        matrices.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """
        # Observations should be tensor, not dictionary
        assert isinstance(observations, torch.Tensor)
        observations = cast(torch.Tensor, observations)
        N = observations.shape[0]

        # Compute/return predicted state and uncertainty values
        # Note that for square C_pinv matrices, we can compute scale_tril as C_pinv @
        # R_tril. In the general case, we transform the full covariance and then take
        # the cholesky decomposition.
        predicted_states = (self.C_pinv[None, :, :] @ observations[:, :, None]).squeeze(-1)
        scale_tril = torch.cholesky(
            self.C_pinv @ self.R_tril @ self.R_tril.transpose(-1, -2) @ self.C_pinv.transpose(-1, -2)
        )[None, :, :].expand((N, self.state_dim, self.state_dim))

        # wrap angular dimensions if they exist
        # if self.ang_dims is not None:
        #     predicted_states[:, self.ang_dims] = wrap_to_pi(predicted_states[:, self.ang_dims])

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        return predicted_states, scale_tril


def get_trainable_model_error(model) -> float:
    """Get the error of our toy trainable models, which all output the correct output +
    a trainable output bias.

    Returns:
        float: Error. Computed as the absolute value of the output bias.
    """

    # Check model validity -- see above for implementation details
    assert hasattr(model, "trainable")
    assert model.trainable is True
    assert hasattr(model, "output_bias")
    assert model.output_bias.shape == (1,)

    # The error is just absolute value of our scalar output bias
    return abs(float(model.output_bias[0]))
