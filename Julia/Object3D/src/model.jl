cam = Camera(5π/16, cam_pose_looking_at(SVector(0, 0, 0), CAMERA_POSITION), 5, 0.4)
num_noisy_samples = 200
noise = 1e-2
@gen function model(T::Int)
    y = 0.0
    for t=1:T
        y = {:y => t} ~ normal(y, DRIFT_VARIANCE)
        {:noisy_pc => t} ~ noisy_renderer(make_scene(0., y, 0.), cam, num_noisy_samples, noise)
    end
end
