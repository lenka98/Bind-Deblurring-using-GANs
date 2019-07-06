from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

from tflearn.layers.conv import global_avg_pool
from vgg19 import VGG

################################## Arguments ##########################

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--content_weight", type=float, default=100.0, help="weight on perceptual term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--scale_size_h", type=int, default=720, help="scale image height to this size before cropping to 256x256")
parser.add_argument("--scale_size_w", type=int, default=1280, help="scale image width to this size before cropping to 256x256")

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_content, gen_grads_and_vars, train")
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")

a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

################################ Utils ########################################

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

            # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size_h, a.scale_size_w], method=tf.image.ResizeMethod.AREA)

        if a.mode == "train":
            offset_h = tf.cast(tf.floor(tf.random_uniform([1], 0, a.scale_size_h - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            offset_w = tf.cast(tf.floor(tf.random_uniform([1], 0, a.scale_size_w - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            if a.scale_size_h > CROP_SIZE and a.scale_size_w > CROP_SIZE:
                r = tf.image.crop_to_bounding_box(r, offset_h[0], offset_w[0], CROP_SIZE, CROP_SIZE)
            else:
                raise Exception("scale size cannot be less than crop size")

        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

####################### Operations ############################################

def relu(x):
    return tf.nn.relu(x)

def lrelu(x):
    return tf.nn.leaky_relu(x)

def bn(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def conv(x, out_channels, kernel = 4, stride = 2, padding = "SAME", scope = "conv"):
    weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    weight_regularizer = None

    with tf.variable_scope(scope):
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], out_channels], initializer=weight_init, regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
        bias = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

    return x

def deconv(x, out_channels, kernel = 4, stride = 2, padding = "SAME", scope = "deconv"):
    weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    weight_regularizer = None
    x_shape = x.get_shape().as_list()
    output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, out_channels]

    with tf.variable_scope(scope):
        w = tf.get_variable("kernel", shape=[kernel, kernel, out_channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
        x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
        bias = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

    return x

def ch_attn(input, channels, scope):
    with tf.variable_scope(scope):
        squeeze = global_avg_pool(input)
        squeeze = tf.reshape(squeeze, [-1, 1, 1, channels])
        squeeze = conv(squeeze, channels//8, kernel = 1, stride = 1, scope = "squeeze_conv")
        excite = relu(squeeze)
        excite = conv(excite, channels, kernel = 1, stride = 1,  scope = "excite_conv")
        attn = tf.nn.sigmoid(excite)
        output = attn * input

    return output

def res(input, channels, scope):
    with tf.variable_scope(scope):
        x = conv(input, channels, stride = 1, scope = "conv1")
        x = ch_attn(x, channels, "ch_attn1")
        x = conv(x, channels, stride = 1, scope = "conv2")
        x = ch_attn(x, channels, "ch_attn2")
        output = input + x

    return output

def encode (input, in_channels, scope):
    with tf.variable_scope(scope):
        with tf.variable_scope("path1"):
            p1 = res(input, in_channels, "res1")
            p1 = res(p1, in_channels, "res2")
            p1 = conv(p1, in_channels * 2)
            p1 = bn(p1)
            p1 = relu(p1)

        with tf.variable_scope("path2"):
            p2 = conv(input, in_channels * 2)
            p2 = bn(p2)
            p2 = relu(p2)

        output = p1 + p2

    return output

def decode (input, in_channels, factor, scope):
    with tf.variable_scope(scope):
        with tf.variable_scope("path1"):
            p1 = res(input, in_channels, "res1")
            p1 = res(p1, in_channels, "res2")
            p1 = deconv(p1, in_channels // factor)
            p1 = bn(p1)
            p1 = relu(p1)
        with tf.variable_scope("path2"):
            p2 = deconv(input, in_channels // factor)
            p2 = bn(p2)
            p2 = relu(p2)

        output = p1 + p2

    return output

def sp_attn (input, channels):
    f = conv(input, channels//8, kernel = 1, stride = 1, scope = "f")
    g = conv(input, channels//8, kernel = 1, stride = 1, scope = "g")
    h = conv(input, channels, kernel = 1, stride = 1, scope = "h")

    attn = tf.nn.softmax(tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b = True))

    v = tf.matmul(attn, hw_flatten(h))
    v_reshaped = tf.reshape(v, shape=input.shape)
    o = conv(v_reshaped, channels, kernel = 1, stride = 1, scope = "v")

    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    output = o * gamma + input

    return output

########################## Network ###########################################

def generator (input_image):

    with tf.variable_scope("en1"):
        en1 = conv(input_image, 64) #[bs, 256, 256, 3] -> [bs, 128, 128, 64]
        en1 = bn(en1)
        en1 = relu(en1)

    with tf.variable_scope("encoder"):
        en2 = encode(en1, 64, "en2") #[bs, 128, 128, 64] -> [bs, 64, 64, 128]
        en3 = encode(en2, 128, "en3") #[bs, 64, 64, 128] -> [bs, 32, 32, 256]
        en4 = encode(en3, 256, "en4") #[bs, 32, 32, 256] -> [bs, 16, 16, 512]
        en5 = encode(en4, 512, "en5") #[bs, 16, 16, 512] -> [bs, 8, 8, 1024]

    with tf.variable_scope("spatial_attention"):
        encoded = sp_attn(en5, 1024) #[bs, 8, 8, 1024] -> [bs, 8, 8, 1024]

    with tf.variable_scope("decoder"):
        de5 = decode(encoded, 1024, 2, "de5") #[bs, 8, 8, 1024] -> [bs, 16, 16, 512]
        de4 = decode(tf.concat([de5, en4], axis = 3), 1024, 4, "de4") #[bs, 16, 16, 1024] -> [bs, 32, 32, 256]
        de3 = decode(tf.concat([de4, en3], axis = 3), 512, 4, "de3") #[bs, 32, 32, 512] -> [bs, 64, 64, 128]
        de2 = decode(tf.concat([de3, en2], axis = 3), 256, 4, "de2") #[bs, 64, 64, 256] -> [bs, 128, 128, 64]
    with tf.variable_scope("de1"):
        de1 = deconv(tf.concat([de2, en1], axis = 3), 3) #[bs, 128, 128, 128] -> [bs, 256, 256, 3]
        de1 = bn(de1)
        decoded = tf.tanh(de1)

    with tf.variable_scope("global_skip"):
        output = (decoded + input_image)/2

    return output

def discriminator(input_image, target_image):
    input = tf.concat([input_image, target_image], axis=3)

    with tf.variable_scope("block1"):
        convolved = conv(input, 64)
        normalized = bn(convolved)
        rectified = lrelu(normalized)

    with tf.variable_scope("block2"):
        convolved = conv(rectified, 128)
        normalized = bn(convolved)
        rectified = lrelu(normalized)

    with tf.variable_scope("block3"):
        convolved = conv(rectified, 256)
        normalized = bn(convolved)
        rectified = lrelu(normalized)

    with tf.variable_scope("block4"):
        convolved = conv(rectified, 512, stride = 1)
        normalized = bn(convolved)
        rectified = lrelu(normalized)

    with tf.variable_scope("block5"):
        convolved = conv(rectified, 1, kernel = 3, stride = 1, padding = "VALID")
        patch = tf.sigmoid(convolved)

    return patch

def create_model(inputs, targets):

    with tf.variable_scope("generator"):
        outputs = generator(inputs)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_content =  tf.reduce_mean(tf.abs(targets - outputs))

        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_content * a.content_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_content])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_content=ema.average(gen_loss_content),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

##################################### Main ####################################

def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_content", model.gen_loss_content)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_content"] = model.gen_loss_content

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_content", results["gen_loss_content"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
