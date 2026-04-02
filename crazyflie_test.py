import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/bitcraze_crazyflie_2/scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        data.ctrl[:] = 0.05  # Hover thrust constant
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Real-time synchronization
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
        
echo "# drone-project-better" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/PsiPJ/drone-project-better.git
git push -u origin main