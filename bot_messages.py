messages = {'nst': ('During the next steps you will be asked to '
                    'provide all the necessary inputs for image style '
                    'transfer.\n\nLets start from content image.\n'
                    'Please, send to me a content image as '
                    'photo attachment.\n\nNote that if image is large '
                    'then it will be resized down so that the '
                    'greater of height and width will be set to 256 '
                    'pixels and other dimension will be '
                    'reduced proportionally.'),
            'info': ("Hi!\nI'm DLS Bot for Neural Style Transfer!\n"
                     "Use /help to get list of available commands."),
            'help': ('In order to perform style transfer for your '
                     'image type /nst command and follow the bot '
                     'guidance.\n\nType /cancel command at any of the '
                     'steps to quit the session.\n\nType /info command '
                     'to read about me.\n\nType /example to see an '
                     'example of usage.'),
            'example': ('These photo show the example image style '
                        'transfer that performs the bot.\n\nThus, '
                        'style is transferred from the mosaic image '
                        '(style image) to the bear image (content '
                        'image) producing the output image.'),
            'content_img_wrong_inp': ('Your message doesn\'t have '
                                      'image attached.\n\nPlease, send '
                                      'content image as photo attachment.\n\n'
                                      'Type /cancel if you\'d like '
                                      'to quit the session.'),
            'style_img_wrong_inp': ('Your message doesn\'t have '
                                    'image attached.\n\nPlease, send '
                                    'style image as photo attachment.\n\n'
                                    'Type /cancel if you\'d like '
                                    'to quit the session.'),
            'style_img_inp': ('Content image is accepted.\n\n'
                              'Now, please, send style image '
                              'as photo attachment'),
            'gamma_inp': ('Style image is accepted.\n\n'
                          'Please set gamma parameter within range (0,10].\n'
                          'Type \'d\' to use default value of 1.0.'),
            'gamma_wrong_inp': ('You set the wrong value for gamma parameter. '
                                 'It has to be within range (0,10] or '
                                 '\'d\' for default value of 1.0.\n\n'
                                 'Type /cancel if you\'d like '
                                 'to quit the session.'),
            'gamma_is_set': 'Gamma parameter is set successfully.',
            'processing': ('Please, wait... Magic is happening.\n'
                           '(image processing will take a few minutes)'),
            'inputs_received': ('All inputs received.\nStyle '
                                'transfer is about to start.'),
            'finish': 'Voila! Here is the resulted image.',
            'echo': ('I\'m sorry.\nMy responses are limited.\n'
                     'You must type the right commands.\n\n'
                     'Use /help command for more details.'),
            'end_session': 'Session has ended.'
            }