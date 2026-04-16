import numpy as np

# ============================================================
# 🔥 FIRE OBJECT MODEL (PURE SIMULATION LOGIC)
# ============================================================
# This does NOT exist in MuJoCo physics.
# It is tracked separately in Python.

class Fire:
    def __init__(self, position):
        self.position = np.array(position)

        # Fire "health" → how alive the fire is
        # 1.0 = fully burning
        # 0.0 = extinguished
        self.intensity = 1.0

        # How fast fire goes out when drone is above it
        self.extinguish_rate = 0.01

    def update(self, drone_pos):
        """
        Called every simulation step.
        Checks if drone is close enough to extinguish fire.
        """

        distance = np.linalg.norm(drone_pos - self.position)

        # =====================================================
        # 🔥 EXTINCTION CONDITION
        # =====================================================
        # If drone is above fire AND close enough → reduce intensity

        if distance < 0.3 and drone_pos[2] > 0.3:
            self.intensity -= self.extinguish_rate

        # Clamp intensity so it never goes negative
        self.intensity = max(0.0, self.intensity)

    def is_out(self):
        """Returns True if fire is extinguished."""
        return self.intensity <= 0.0