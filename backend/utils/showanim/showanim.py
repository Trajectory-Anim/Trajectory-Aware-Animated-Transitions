from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import random
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

def get_color(index, isTransparent=False, alpha=0.1):
    index[0] = index[0] % 10
    colors = plt.cm.get_cmap('tab10')
    color = colors(index)
    color = (color * 255)[0]
    if isTransparent:
        color = alpha * color + (1 - alpha) * 255
    color = (int(color[2]), int(color[1]), int(color[0]))
    return color

def show_animation(
        anime_position,
        cluster,
        anime_start,
        anime_end,
        radius,
        width=1024,
        height=1024,
        title='Scatter Plot Transition Animation',
        fps=30,
        margin_percentage=0.05,
        fade_time=0.5,
        save_to_file_name='scatter_plot_transition_animation.mp4',
        resolution_scale=1,
        bundled_position=None,
        # radius=10/1024,
    ):
    width *= resolution_scale
    height *= resolution_scale

    # circle_size = int(radius * width) - 1

    if bundled_position is None:
        bundled_position = anime_position
    elif len(bundled_position) == 0:
        bundled_position = anime_position
    
    print(f'shape of anime_position: {anime_position.shape}')
    print(f'shape of bundled_position: {bundled_position.shape}')
    N = anime_position.shape[1]

    amount_frames = anime_position.shape[0]
    fade_frames = int(fade_time * fps)
    N_clusters = cluster.shape[1]
    start_of_clusters = np.ones(N_clusters, dtype=int)*int(1e9)
    end_of_clusters = np.ones(N_clusters, dtype=int)*int(-1e9)
    for clusteridx in range(N_clusters):
        cluster_points = []
        for i in range(N):
            if cluster[i, clusteridx] == 1:
                cluster_points.append(i)
        for i in range(len(cluster_points)):
            start_of_clusters[clusteridx] = min(start_of_clusters[clusteridx], anime_start[cluster_points[i]])
            end_of_clusters[clusteridx] = max(end_of_clusters[clusteridx], anime_end[cluster_points[i]])

    # print(f'start_of_clusters: {start_of_clusters}')
    # print(f'end_of_clusters: {end_of_clusters}')
    # print(f'amount_frames: {amount_frames}')
    # sort the clusters by start time
    cluster_order = list(range(N_clusters))
    cluster_order.sort(key=lambda x: -start_of_clusters[x])

    frame_to_copy = start_of_clusters.tolist() + end_of_clusters.tolist()
    frame_to_copy = list(set(frame_to_copy))
    # print(f'frame_to_copy: {frame_to_copy}')

    # add fade in and fade out times for each cluster
    # first we need to insert extra fade_frames into the anime_position and bundled_position

    def copy_frame(frame_id, copy_num):
        nonlocal amount_frames, anime_start, anime_end, start_of_clusters, end_of_clusters, anime_position, bundled_position, frame_to_copy
        if frame_id >= amount_frames:
            raise ValueError(f'frame_id {frame_id} is larger than amount_frames {amount_frames}')
        cur_anime_pos = anime_position[frame_id]
        cur_bundled_pos = bundled_position[frame_id]

        # insert the frame into the anime_position and bundled_position
        for i in range(copy_num):
            anime_position = np.insert(anime_position, frame_id, cur_anime_pos, axis=0)
            bundled_position = np.insert(bundled_position, frame_id, cur_bundled_pos, axis=0)

        # affect the frame id behind
        amount_frames += copy_num
        for i in range(N):
            if anime_start[i] > frame_id:
                anime_start[i] += copy_num
            if anime_end[i] > frame_id:
                anime_end[i] += copy_num
        for i in range(N_clusters):
            if start_of_clusters[i] > frame_id:
                start_of_clusters[i] += copy_num
            if end_of_clusters[i] > frame_id:
                end_of_clusters[i] += copy_num
        for i in range(len(frame_to_copy)):
            if frame_to_copy[i] > frame_id:
                frame_to_copy[i] += copy_num

    for frame_id in frame_to_copy:
        copy_frame(frame_id, fade_frames)

    def get_image_x(origin_x):
        nonlocal width
        nonlocal margin_percentage
        origin_x = origin_x * (1 - 2 * margin_percentage) + margin_percentage
        return int (origin_x * width)
    
    def get_image_y(origin_y):
        nonlocal height
        nonlocal margin_percentage
        origin_y = origin_y * (1 - 2 * margin_percentage) + margin_percentage
        origin_y = 1 - origin_y
        return int (origin_y * height)

    def get_single_image(timepoint):
        nonlocal anime_position
        nonlocal bundled_position
        nonlocal anime_start
        nonlocal anime_end

        positions = anime_position[timepoint]
        image = 255 * np.ones((height, width, 3), np.uint8)

        # draw each point
        draw_order = list(range(positions.shape[0]))
        # sort by anime_end
        def sort_key(x):
            if timepoint > anime_end[x]:
                return 1e5 + anime_end[x]
            else:
                return anime_end[x]
        draw_order.sort(key=sort_key, reverse=True)
        # draw the trace of the moving points
        for i in draw_order:
            clusteridx = np.where(cluster[i] == 1)[0]
            circle_size = int(radius[i] * width) - 1

            if timepoint >= start_of_clusters[clusteridx]:
                for each_time in range(anime_start[i], anime_end[i]):

                    if timepoint >= each_time:
                        color = (235, 235, 235)
                    else:
                        continue

                    tx1 = get_image_x(bundled_position[each_time][i][0])
                    ty1 = get_image_y(bundled_position[each_time][i][1])
                    tx2 = get_image_x(bundled_position[each_time + 1][i][0])
                    ty2 = get_image_y(bundled_position[each_time + 1][i][1])

                    cv2.line(image, (tx1, ty1), (tx2, ty2), color, circle_size * 2 + 1, lineType=cv2.LINE_AA)
        
        for i in draw_order:
            clusteridx = np.where(cluster[i] == 1)[0]
            
            color_end = get_color(clusteridx, isTransparent=True)
            position = positions[i]

            x = get_image_x(position[0])
            y = get_image_y(position[1])

            isEnded = (timepoint >= anime_end[i] + fade_frames)

            circle_size = int(radius[i] * width) - 1
            if isEnded:
                cv2.circle(image, (x, y), circle_size, color_end, -1, lineType=cv2.LINE_AA)
        
        for i in draw_order:
            clusteridx = np.where(cluster[i] == 1)[0]
            circle_size = int(radius[i] * width) - 1

            if timepoint >= start_of_clusters[clusteridx]:
                for each_time in range(anime_start[i], anime_end[i]):

                    if timepoint >= each_time:
                        continue;
                    else:
                        color = (200, 200, 200)

                    tx1 = get_image_x(bundled_position[each_time][i][0])
                    ty1 = get_image_y(bundled_position[each_time][i][1])
                    tx2 = get_image_x(bundled_position[each_time + 1][i][0])
                    ty2 = get_image_y(bundled_position[each_time + 1][i][1])

                    cv2.line(image, (tx1, ty1), (tx2, ty2), color, circle_size * 2 + 1, lineType=cv2.LINE_AA)

        for i in draw_order:
            clusteridx = np.where(cluster[i] == 1)[0]
            
            color_end = get_color(clusteridx, isTransparent=True)
            position = positions[i]

            x = get_image_x(position[0])
            y = get_image_y(position[1])

            isEnded = (timepoint >= anime_end[i] + fade_frames)
            isStarted = (timepoint >= start_of_clusters[clusteridx])

            circle_size = int(radius[i] * width) - 1
            if isEnded:
                continue

            fade_in_start = start_of_clusters[clusteridx]
            fade_out_start = anime_end[i]

            # if not started, fade_ratio = 0
            if not isStarted:
                fade_ratio = 0
            # if is fading in
            elif timepoint >= fade_in_start and timepoint <= fade_in_start + fade_frames:
                fade_ratio = (timepoint - fade_in_start) / (fade_frames)
            # if is fading out
            elif timepoint >= fade_out_start and timepoint <= fade_out_start + fade_frames:
                fade_ratio = (fade_out_start + fade_frames - timepoint) / (fade_frames)
            else:
                fade_ratio = 1

            color_border = get_color(clusteridx, isTransparent=True, alpha=fade_ratio)

            if fade_ratio > 0:
                cv2.circle(image, (x, y), circle_size, color_border, -1, lineType=cv2.LINE_AA)
        
        cv2.putText(image, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color([0]), 1, lineType=cv2.LINE_AA)
                
        return image

    image_list = []
    # use pool to speed up the computation
    with Pool() as pool:
        image_list = list(tqdm(pool.imap(get_single_image, range(amount_frames)), total=amount_frames, desc="Computing images"))

    save_animation_to_file(save_to_file_name, image_list, fps=fps)

def save_animation_to_file(save_to_file_name, image_list, fps):
    # save the images into a mp4 video
    image_list_RGB = []
    for timepoint in tqdm(range(len(image_list)), desc="Saving video    "):
        image = image_list[timepoint]
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list_RGB.append(converted_image)

    imageio.mimsave('./log/video/' + save_to_file_name, image_list_RGB, fps=fps)
    print(f"Saved to {save_to_file_name}")
