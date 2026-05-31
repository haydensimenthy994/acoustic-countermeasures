"""
PATCH for src/attacks/pgd.py

Add this block INSIDE your PGD evaluation loop — wherever you currently
aggregate results. Saves per-sample clean/adv confidences AND one example
clean+adversarial waveform for use by figures 4 and 6.

INTEGRATION NOTES:
  Your existing evaluate() function presumably iterates over the test set
  producing (clean_audio, adv_audio, clean_logits, adv_logits, labels).
  Inside that loop, collect them. After the loop, dump a .npz.

Drop this file in as src/attacks/_pgd_logging.py and call from pgd.py.
"""
import numpy as np
import torch
from pathlib import Path


class PGDSampleLogger:
    """
    Collects per-sample data during PGD evaluation, saves one .npz at the end.

    Usage inside your PGD evaluate() function:

        logger = PGDSampleLogger(epsilon=eps)
        for batch in loader:
            clean = batch["waveform"]
            label = batch["label"]
            adv   = pgd_attack(model, clean, label, eps, ...)

            with torch.no_grad():
                clean_probs = model(clean.cuda()).softmax(-1).cpu()
                adv_probs   = model(adv.cuda()  ).softmax(-1).cpu()

            logger.add(
                clean_audio=clean, adv_audio=adv,
                clean_probs=clean_probs, adv_probs=adv_probs,
                labels=label,
            )

        logger.save(f"outputs/results/pgd_samples_eps{eps}.npz")
    """
    def __init__(self, epsilon, keep_one_example=True):
        self.epsilon = epsilon
        self.keep_one_example = keep_one_example
        self.clean_conf_correct = []   # P(correct class) before attack
        self.adv_conf_correct   = []   # P(correct class) after attack
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

        # Save the first drone-class sample that was successfully flipped
        if self.keep_one_example and self.example_clean is None:
            for i, y in enumerate(labels):
                was_correct = clean_probs[i].argmax() == y
                now_wrong   = adv_probs  [i].argmax() != y
                if was_correct and now_wrong and y == 1:  # drone -> no_drone
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
