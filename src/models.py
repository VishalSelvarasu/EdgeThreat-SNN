import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

import config


def encode(x, time_steps, method="rate", generator=None):
    """(batch, features) in [0, 1] -> (time_steps, batch, features)."""
    if method == "rate":     # each value is the spike probability at every step
        p = x.clamp(0.0, 1.0)
        return (torch.rand((time_steps, *p.shape), generator=generator) < p).float()
    # the value itself as input current (used in the ablation)
    if method == "direct":
        return x.unsqueeze(0).expand(time_steps, *x.shape)
    raise ValueError(f"unknown encoding {method!r}")


class MLP(nn.Module):
    """hidden=(64, 32) is the baseline; hidden=(64,) has exactly the SNN's parameter count."""

    def __init__(self, input_dim, hidden=(64, 32), output_dim=2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, generator=None):   # same signature as the SNN
        return self.net(x)


class EdgeThreatSNN(nn.Module):
    """Encoding -> 64 LIF neurons (threshold 1.0, reset by subtraction, fast-sigmoid
    surrogate) -> 2 non-spiking leaky integrators whose mean membrane potential is the logit."""

    def __init__(self, input_dim, hidden=config.HIDDEN, output_dim=2, beta=config.BETA,
                 time_steps=config.TIME_STEPS, encoding=config.ENCODING):
        super().__init__()
        self.time_steps = time_steps
        self.encoding = encoding
        self.fc1 = nn.Linear(input_dim, hidden)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.fc2 = nn.Linear(hidden, output_dim)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")
        self.record_stats = False
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"samples": 0, "input_events": 0.0, "hidden_spikes": 0.0}

    def forward(self, x, generator=None):
        inputs = encode(x, self.time_steps, self.encoding, generator)
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        total, hidden_spikes = 0.0, 0.0
        for t in range(self.time_steps):
            spk1, mem1 = self.lif1(self.fc1(inputs[t]), mem1)
            _, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + mem2
            if self.record_stats:
                hidden_spikes += float(spk1.detach().sum())
        if self.record_stats:
            self.stats["samples"] += x.shape[0]
            self.stats["input_events"] += float((inputs != 0).sum())
            self.stats["hidden_spikes"] += hidden_spikes
        return total / self.time_steps


NEURAL_KINDS = ("mlp", "mlp_small", "snn")
SKLEARN_KINDS = ("logreg", "rf")
DISPLAY_NAMES = {"mlp": "MLP (64-32)", "mlp_small": "MLP (64), size-matched",
                 "snn": "EdgeThreat-SNN", "logreg": "Logistic regression", "rf": "Random forest"}


def build_model(kind, input_dim, **snn_kwargs):
    if kind == "mlp":
        return MLP(input_dim, hidden=(config.HIDDEN, 32))
    if kind == "mlp_small":
        return MLP(input_dim, hidden=(config.HIDDEN,))
    if kind == "snn":
        return EdgeThreatSNN(input_dim, **snn_kwargs)
    raise ValueError(f"unknown neural model kind {kind!r}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
