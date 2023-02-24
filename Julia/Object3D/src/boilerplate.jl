using StaticArrays: SVector, MMatrix
using CoordinateTransformations
using GeometryBasics: HyperSphere, HyperRectangle, Vec, Point
using ColorTypes
using Rotations
using Gen
using MeshCat
using DifferentiableRenderer: Shape, Parallelogram, unit_pose, Pose, Camera,
                              cam_pose_looking_at, render_point_cloud, Scene,
                              BaseSGNode

#const CAMERA_POSITION = SVector(2, -3, 1.4)
const CAMERA_POSITION = SVector(4, 0, 1.4)
const CUBE_SIZE = 0.5
const DRIFT_VARIANCE = 0.7
const PC_VIZ_RADIUS = 5e-3

vis = nothing

function init_visualizer()
    global vis
    vis = Visualizer()
    settransform!(vis["/Cameras/default"], Translation(0, 0, 0))
    setprop!(vis["/Cameras/default/rotated/<object>"], "position",
             [CAMERA_POSITION[1], CAMERA_POSITION[3], -CAMERA_POSITION[2]])
    setprop!(vis["/Background"], "top_color", RGB(1, 1, 1))
    setprop!(vis["/Background"], "bottom_color", RGB(1, 1, 1))
    open(vis)
end

function reset_visualizer()
    global vis
    delete!(vis)
end


function make_cube()
    xdir, ydir = CUBE_SIZE .* (SVector(1, 0), SVector(0, 1))
    xshift, yshift, zshift = CUBE_SIZE .* (SVector(1, 0, 0), SVector(0, 1, 0),
                                           SVector(0, 0, 1))
    z = zero(SVector{3})
    cube = Shape([Parallelogram(xdir, ydir, unit_pose),
                  Parallelogram(xdir, ydir, Pose(zshift, RotX(0))),
                  Parallelogram(xdir, ydir, Pose(z, RotX(π/2))),
                  Parallelogram(xdir, ydir, Pose(yshift, RotX(π/2))),
                  Parallelogram(xdir, ydir, Pose(z, RotY(-π/2))),
                  Parallelogram(xdir, ydir, Pose(xshift, RotY(-π/2)))])
end


function visualize_state(tr)
    meshcat_cube = HyperRectangle(Vec(0., 0, 0), Vec(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    setobject!(vis[:cube], meshcat_cube)
    setprop!(vis[:cube], "color", RGB(0, 0.5, 0))
    T = get_args(tr)[1]
    ys = [0, map(t-> tr[:y => t], 1:T)...]
    for y in ys 
        settransform!(vis[:cube], Translation(0, y, 0))
        save_image(vis)
        sleep(1)
    end
end

function visualize_pc(pc)
    delete!(vis["pc"])
    for i=1:size(pc)[2]
        setobject!(vis["pc/$i"], HyperSphere(Point(pc[:,i]), PC_VIZ_RADIUS))
        setprop!(vis["pc/$i"], "color", RGB(0, 0, 0))
    end
end

the_cube = make_cube()
make_scene(x::Float64, y::Float64, z::Float64) = 
    Scene([BaseSGNode(the_cube, Pose(SVector(x, y, z), RotX(0)))])

function visualize_renderings(tr)
    T = get_args(tr)[1]
    for t=1:T
        visualize_pc(tr[:noisy_pc => t])
        save_image(vis)
        sleep(1)
    end
end

function visualize_inference(particles, weights, obs)
    T, N = length(obs), length(particles[1])
    meshcat_cube = HyperRectangle(Vec(0., 0, 0), Vec(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    for t=1:T
        delete!(vis["pc"])
        visualize_pc(obs[t])
        alphas = 0.8 .* exp.(weights[t] .- logsumexp(weights[t]))
        for i=1:N
            delete!(vis["particle_$i"])
            setobject!(vis["particle_$i"], meshcat_cube)
            settransform!(vis["particle_$i"], Translation(0, particles[t][i], 0))
            setprop!(vis["particle_$i"], "color", RGBA(0, 0.5, 0, alphas[i]))
            sleep(1)
        end
        sleep(1)
        save_image(vis)
    end
end
