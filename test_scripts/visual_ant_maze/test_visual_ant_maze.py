from test_scripts.utils import save_image, generate_video
from rlnav.visual_ant_maze import VisualAntMaze, VisualAntMazeMapsIndex


environment = VisualAntMaze(maze_name=VisualAntMazeMapsIndex.MEDIUM.value)
environment.reset()

generated_images = []
full_images = []
ant_images = []
pov_images = []


for step_id in range(50):
    action = environment.action_space.sample()
    environment.step(action)

    generated_images.append(environment.render(mode="generated"))
    save_image(generated_images[-1], output_directory="render_generated", file_name="generated_" + str(step_id))
    full_images.append(environment.render(mode="full"))
    save_image(full_images[-1], output_directory="render_full", file_name="full_" + str(step_id))
    ant_images.append(environment.render(mode="ant"))
    save_image(ant_images[-1], output_directory="render_ant", file_name="ant_" + str(step_id))
    pov_images.append(environment.render(mode="pov"))
    save_image(pov_images[-1], output_directory="render_pov", file_name="pov_" + str(step_id))

generate_video(generated_images, filename="generated_images")
generate_video(full_images, filename="full_images")
generate_video(ant_images, filename="ant_images")
generate_video(pov_images, filename="pov_images")

debug = 1