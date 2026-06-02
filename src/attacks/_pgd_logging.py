"""Per-sample logger used inside the PGD eval loop.

Feeds figures 4 (confidence histogram) and 6 (clean-vs-adv waveform pair).
"""
import numpy as np
import torch
from pathlib import Path


class PGDSampleLogger:
    """Collect per-sample confidences during PGD eval, dump one .npz at the end."""
    def __init__(self, epsilon, keep_one_example=True):
        self.epsilon = epsilon
        self.keep_one_example = keep_one_example
        self.clean_conf_correct = []
        self.adv_conf_correct   = []
        self.predicted_before   = []
        self.predicted_after    = []
        self.labels             = []
        self.example_clean      = None
        self.example_adv        = None
        self.example_label      = None

    def add(self, clean_audio, adv_audio, clean_probs, adv_probs, labels):
        labels      = labels.cpu().numpy()
        clean_probs = clean_probs.cpu().numpy() if torch.is_tensor(clean_probs) else clean_probs
        adv_probs   = adv_probs  .cpu().numpy() if torch.is_tensor(adv_probs)   else adv_probs

        for i, y in enumerate(labels):
            self.clean_conf_correct.append(float(clean_probs[i, y]))
            self.adv_conf_correct  .append(float(adv_probs  [i, y]))
            self.predicted_before.append(int(clean_probs[i].argmax()))
            self.predicted_after .append(int(adv_probs  [i].argmax()))
            self.labels          .append(int(y))

        # Hang on to the first drone clip the attack successfully flipped — fig 6 plots it.
        if self.keep_one_example and self.example_clean is None:
            for i, y in enumerate(labels):
                was_correct = clean_probs[i].argmax() == y
                now_wrong   = adv_probs  [i].argmax() != y
                if was_correct and now_wrong and y == 1:
                    self.example_clean = clean_audio[i].detach().cpu().numpy()
                    self.example_adv   = adv_audio  [i].detach().cpu().numpy()
                    self.example_label = int(y)
                    break

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            epsilon=self.epsilon,
            clean_conf_correct=np.array(self.clean_conf_correct),
            adv_conf_correct=np.array(self.adv_conf_correct),
            predicted_before=np.array(self.predicted_before),
            predicted_after=np.array(self.predicted_after),
            labels=np.array(self.labels),
            example_clean=self.example_clean
                if self.example_clean is not None else np.array([]),
            example_adv=self.example_adv
                if self.example_adv is not None else np.array([]),
            example_label=self.example_label
                if self.example_label is not None else -1,
        )
        print(f"saved {len(self.labels)} samples -> {path}")
