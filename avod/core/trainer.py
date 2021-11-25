"""Detection model trainer.

This file provides a generic training method to train a
DetectionModel.
"""
import datetime
import os
import tensorflow as tf
import time

from avod.builders import optimizer_builder
from avod.core import trainer_utils
from avod.core import summary_utils

from utils_sin.sin_utils import SINFields

slim = tf.contrib.slim

def _get_variables_to_train(trainable_scopes_list=None):
    """Returns a list of variables to train.
    Returns:
    A list of variables to train by the optimizer.
    """
    if len(trainable_scopes_list) == 0:
        return tf.compat.v1.trainable_variables()

    variables_to_train = []
    for scope in trainable_scopes_list:
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def train(model, train_config):
    """Training function for detection models.

    Args:
        model: The detection model object.
        train_config: a train_*pb2 protobuf.
            training i.e. loading RPN weights onto AVOD model.
    """

    # Check SIN related configs
    pretrained_ckpt = None
    if train_config.do_train_both and train_config.do_train_sin \
        and train_config.do_train_sin_alt and train_config.do_train_sin_fast:
        raise ValueError(
            'Train SIN and Both noise mode cannot be True at the same time.')
    else:
        # Pretrained_ckpt is only used for handling noise in training
        if train_config.do_train_both or train_config.do_train_sin \
            or train_config.do_train_sin_alt or train_config.do_train_sin_fast:
            if len(train_config.pretrained_ckpt) != 0:
                # Default value is blank string, ""
                pretrained_ckpt = train_config.pretrained_ckpt

    model = model
    train_config = train_config
    # Get model configurations
    model_config = model.model_config

    # Create a variable tensor to hold the global step
    global_step_tensor = tf.Variable(
        0, trainable=False, name='global_step')

    #############################
    # Get training configurations
    #############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = \
        train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    paths_config = model_config.paths_config
    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    checkpoint_dir = paths_config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/' + \
        model_config.checkpoint_name

    global_summaries = set([])

    # The model should return a dictionary of predictions
    prediction_dict = model.build()

    summary_histograms = train_config.summary_histograms
    summary_img_images = train_config.summary_img_images
    summary_bev_images = train_config.summary_bev_images

    ##############################
    # Setup loss
    ##############################
    losses_dict, total_loss = model.loss(prediction_dict)

    # Optimizer
    training_optimizer = optimizer_builder.build(
        train_config.optimizer,
        global_summaries,
        global_step_tensor)

    # Variables to train (Fine-tuning)
    variables_to_train = _get_variables_to_train(train_config.trainable_scopes_list)

    # Create the train op
    with tf.compat.v1.variable_scope('train_op'):
        train_op = slim.learning.create_train_op(
            total_loss,
            training_optimizer,
            clip_gradient_norm=1.0,
            global_step=global_step_tensor,
            variables_to_train=variables_to_train)

    # Add the result of the train_op to the summary
    tf.compat.v1.summary.scalar("training_loss", train_op)

    # Save checkpoints regularly.
    saver = tf.compat.v1.train.Saver(max_to_keep=max_checkpoints,
                           pad_step_number=True)

    # Add maximum memory usage summary op
    # This op can only be run on device with gpu
    # so it's skipped on travis
    is_travis = 'TRAVIS' in os.environ
    if not is_travis:
        # tf.summary.scalar('bytes_in_use',
        #                   tf.contrib.memory_stats.BytesInUse())
        tf.compat.v1.summary.scalar('max_bytes',
                          tf.contrib.memory_stats.MaxBytesInUse())

    summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(
        summaries,
        global_summaries,
        histograms=summary_histograms,
        input_imgs=summary_img_images,
        input_bevs=summary_bev_images
    )

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
    if allow_gpu_mem_growth:
        # GPU memory config
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = allow_gpu_mem_growth
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()

    # Create unique folder name using datetime for summary writer
    datetime_str = str(datetime.datetime.now())
    logdir = logdir + '/train'
    train_writer = tf.compat.v1.summary.FileWriter(logdir + '/' + datetime_str,
                                         sess.graph)

    # Create init op
    init = tf.compat.v1.global_variables_initializer()

    # Continue from last saved checkpoint
    if not train_config.overwrite_checkpoints:
        if pretrained_ckpt is None:
            trainer_utils.load_checkpoints(checkpoint_dir,
                                           saver)
            if len(saver.last_checkpoints) > 0:
                checkpoint_to_restore = saver.last_checkpoints[-1]
                saver.restore(sess, checkpoint_to_restore)
            else:
                # Initialize the variables
                sess.run(init)
        else:
            # Restore pretrained_ckpt to start from the pretrained model
            saver.restore(sess, pretrained_ckpt)
    else:
        # Initialize the variables
        sess.run(init)

    # Read the global step if restored
    global_step = tf.compat.v1.train.global_step(sess,
                                       global_step_tensor)
    print('Starting from step {} / {}'.format(
        global_step, max_iterations))

    idx_max_loss = 0
    # Main Training Loop
    last_time = time.time()
    for step in range(global_step, max_iterations + 1):

        # Save checkpoint
        if step % checkpoint_interval == 0:
            global_step = tf.compat.v1.train.global_step(sess,
                                               global_step_tensor)

            saver.save(sess,
                       save_path=checkpoint_path,
                       global_step=global_step)

            print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                step, max_iterations,
                checkpoint_path, global_step))

        # Create feed_dict for inferencing
        if train_config.do_train_both:
            # Alternate between noisy and clean data
            if step % 2 == 0:
                feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                   sin_level=train_config.sin_level,
                                                   gen_all_sin_inputs=True)
            else:
                feed_dict = model.create_feed_dict()
        elif train_config.do_train_sin_alt:
            # Alternate between noisy (single-input) and clean data
            if step % 4 == 0:
                feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                   sin_level=train_config.sin_level,
                                                   sin_input_name='image')
            if step % 4 == 2:
                feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                   sin_level=train_config.sin_level,
                                                   sin_input_name='lidar')
            else:
                feed_dict = model.create_feed_dict()
        elif train_config.do_train_sin:
            # Train MaxSIN (direct)
            if step % 2 == 0:
                # Calculate losses per SIN input first
                list_feed_dict = []
                list_total_loss = []
                for sin_input_name in SINFields.SIN_INPUT_NAMES:
                    if sin_input_name == SINFields.SIN_INPUT_NAMES[0]:
                        # First feed_dict is generated normally
                        temp_feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                                sin_level=train_config.sin_level,
                                                                sin_input_name=sin_input_name)
                    else:
                        # As it was visited ocne more, next_batch function has to access previous one
                        temp_feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                                sin_level=train_config.sin_level,
                                                                sin_input_name=sin_input_name,
                                                                get_prev_batch=True)
                    list_feed_dict.append(temp_feed_dict)
                    temp_total_loss = sess.run([total_loss],feed_dict=temp_feed_dict)
                    list_total_loss.append(temp_total_loss)
                # Choose input (feed_dict) with the higher loss
                idx_max_loss = list_total_loss.index(max(list_total_loss))
                feed_dict = list_feed_dict[idx_max_loss]
                # Total loss value at actual train_op process is slightly different
                # because of some randomness in layers (i.e. dropout)
            else:
                feed_dict = model.create_feed_dict()
        elif train_config.do_train_sin_fast:
            # Train MaxSIN (Fast)
            if step % 2 == 0:
                if step % (2*train_config.n_inner_sin_fast):
                    # Calculate losses per SIN input first
                    list_feed_dict = []
                    list_total_loss = []
                    for sin_input_name in SINFields.SIN_INPUT_NAMES:
                        if sin_input_name == SINFields.SIN_INPUT_NAMES[0]:
                            # First feed_dict is generated normally
                            temp_feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                                    sin_level=train_config.sin_level,
                                                                    sin_input_name=sin_input_name)
                        else:
                            # As it was visited ocne more, next_batch function has to access previous one
                            temp_feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                                    sin_level=train_config.sin_level,
                                                                    sin_input_name=sin_input_name,
                                                                    get_prev_batch=True)
                        list_feed_dict.append(temp_feed_dict)
                        temp_total_loss = sess.run([total_loss],feed_dict=temp_feed_dict)
                        list_total_loss.append(temp_total_loss)
                    # Choose input (feed_dict) with the higher loss
                    idx_max_loss = list_total_loss.index(max(list_total_loss))
                    feed_dict = list_feed_dict[idx_max_loss]
                    # Total loss value at actual train_op process is slightly different
                    # because of some randomness in layers (i.e. dropout)
                else:
                    # Use previouly selected input source (argmax of loss) to generate Noise
                    sin_input_name = SINFields.SIN_INPUT_NAMES[idx_max_loss]
                    feed_dict = model.create_feed_dict(sin_type=train_config.sin_type,
                                                       sin_level=train_config.sin_level,
                                                       sin_input_name=sin_input_name)

            else:
                feed_dict = model.create_feed_dict()
        else:
            feed_dict = model.create_feed_dict()

        # Write summaries and train op
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time

            train_op_loss, summary_out = sess.run(
                [train_op, summary_merged], feed_dict=feed_dict)

            print('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                step, train_op_loss, time_elapsed))
            train_writer.add_summary(summary_out, step)
            
            # total_loss_ho = sess.run([total_loss],feed_dict=feed_dict)
            # if train_config.do_train_sin:
            #     print('   !!! Debugging..',list_total_loss,idx_max_loss,train_op_loss)

        else:
            # Run the train op only
            sess.run(train_op, feed_dict)

    # Close the summary writers
    train_writer.close()
