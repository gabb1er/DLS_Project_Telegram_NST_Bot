messages = {'nst': ('During the next steps you will be asked to '
                      'provide all the necessary inputs for image style '
                      'transfer. Lets start from content image. '
                      'Please, send to me a content image as '
                      'photo attachment.'),
            'info': ("Hi!\nI'm DLS Bot for Neural Style Transfer!\n"
                     "Use /help to get list of available commands."),
            'help': ('In order to perform style transfer for your '
                     'image type /nst command and follow the bot '
                     'guidance. Type /cancel command at any of the '
                     'steps to quit the session. Type /info command '
                     'to read about me.'),
            'content_img_wrong_inp': ('Your message doesn\'t have '
                                      'image attached. Please, send '
                                      'content image as photo attachment. '
                                      'Type /cancel if you\'d like '
                                      'to quit the session.'),
            'style_img_wrong_inp': ('Your message doesn\'t have '
                                    'image attached. Please, send '
                                    'style image as photo attachment. '
                                    'Type /cancel if you\'d like '
                                    'to quit the session.'),
            'style_img_inp': ('Content image is accepted. '
                              'Now, please, send style image '
                              'as photo attachment'),
            'gamma_inp': ('Style image is accepted. '
                          'Please set gamma parameter within range (0,10]. '
                          'Type \'d\' to use default value of 1.0.'),
            'gamma_wrong_inp': ('You set the wrong value for gamma parameter. '
                                 'It has to be within range (0,10] or '
                                 '\'d\' for default value of 1.0. '
                                 'Type /cancel if you\'d like '
                                 'to quit the session.'),
            'resize_inp': ('Gamma parameter is set successfully. '
                           'Type \'y\' if you\'d like to resize '
                           'your image. In this case the greater of '
                           'height and width will be set to 256 '
                           'pixels and other dimension will be '
                           'reduced proportionally. Type \'n\' '
                           ' if you\'d like to process image as is. '
                           'Note that processing of large image might '
                           'take a long time.'),
            'resize_wrong_inp': ('Wrong input. Please type \'y\' '
                                 'if you\'d like to resize the image. '
                                 'Otherwise type \'n\'. '
                                 'Type /cancel if you\'d like '
                                 'to quit the session.'),
            'processing': 'Please, wait... Magic is happening.',
            'inputs_received': ('All inputs received. Style '
                                'transfer is about to start.'),
            'finish': 'Voila! Here is the resulted image.',
            'echo': ('I\'m sorry. My responses are limited. '
                     'You must type the right commands. '
                     'Use /help command for more details.'),
            'end_session': 'Session has ended.'
            }