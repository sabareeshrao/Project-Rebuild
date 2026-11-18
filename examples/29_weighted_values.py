"""Lesson 29: aggregate content using a weighted average of value vectors."""

import torch

from rebuild_gpt.attention import weighted_sum_values

weights = torch.tensor([[[1., 0.], [0.25, 0.75]]])
values = torch.tensor([[[2., 0.], [0., 4.]]])
mixed = weighted_sum_values(weights, values)

print("weights [B,T,S]:", tuple(weights.shape))
print("values [B,S,D]:", tuple(values.shape))
print("mixed [B,T,D]:", tuple(mixed.shape))
print("mixed vectors:", mixed.tolist())  # [[[2,0],[0.5,3]]]
