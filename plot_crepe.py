# plot_crepe.py
import numpy as np
import matplotlib.pyplot as plt

# Load saved CREPE outputs
f0 = np.load("melody_f0.npy")        # frequency in Hz (NaN = unvoiced)
conf = np.load("melody_conf.npy")    # CREPE confidence
times = np.load("melody_time.npy")   # timestamps in seconds

plt.figure(figsize=(14, 6))

# -------- F0 plot ----------
plt.subplot(2, 1, 1)
plt.plot(times, f0, linewidth=1)
plt.title("CREPE Estimated F0 (Melody)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 1200)  # adjust if needed
plt.grid(True)

# -------- Confidence plot ----------
plt.subplot(2, 1, 2)
plt.plot(times, conf, color="orange", linewidth=1)
plt.title("CREPE Confidence")
plt.xlabel("Time (s)")
plt.ylabel("Confidence (0–1)")
plt.ylim(0, 1.05)
plt.grid(True)

plt.tight_layout()
plt.show()
