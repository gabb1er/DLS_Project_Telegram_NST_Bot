"""
This module describes TG bot
for style transfer
"""

import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.message import ContentType
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from config import TOKEN
from bot_messages import messages
from style_transfer import image_style_transfer

class RequestOrder(StatesGroup):
    content_inp = State()
    style_inp = State()
    gamma_inp = State()
    resize_inp = State()


# Initialize bot and dispatcher
bot = Bot(token=TOKEN, timeout=600)
mem_storage = MemoryStorage()
dp = Dispatcher(bot, storage=mem_storage)

# Configure logging
logging.basicConfig(level=logging.INFO)


async def run_nst(inp):
    content_img_name = str(inp['user_id']) + '_content_img.jpg'
    style_img_name = str(inp['user_id']) + '_style_img.jpg'
    await bot.download_file_by_id(inp['content_img_id'], destination=content_img_name)
    await bot.download_file_by_id(inp['style_img_id'], destination=style_img_name)
    await image_style_transfer(inp['gamma'], inp['resize'], str(inp['user_id']))


@dp.message_handler(commands=['start'], state="*")
async def process_start_command(message: types.Message):
    """
    This handler will be called when user sends `/start` command
    """
    await message.answer(messages['info'])

@dp.message_handler(commands=['help'], state="*")
async def process_help_command(message: types.Message):
    """
    This handler will be called when user sends `/help` command
    """
    await message.answer(messages['help'])

@dp.message_handler(commands=['info'], state="*")
async def process_info_command(message: types.Message):
    """
    This handler will be called when user sends `/info` command
    """
    await message.answer(messages['info'])

@dp.message_handler(commands=['cancel'], state="*")
async def end_session(message: types.Message,
                      state: FSMContext):
    """
    This handler will be called when user sends /cancel command
    """
    await state.finish()
    await message.answer(messages['end_session'])

@dp.message_handler(commands=['nst'], state="*")
async def start_nst(message: types.Message):
    """
    This handler will be called when user sends /nst commnad.
    Inputs collection dialog is initiated.
    """
    await message.answer(messages['nst'])
    await RequestOrder.content_inp.set()

@dp.message_handler(state=RequestOrder.content_inp, content_types=ContentType.ANY) #=ContentType.PHOTO)
async def get_content_image(message: types.Message,
                            state: FSMContext):
    """
    This handler will be called during inputs collection dialog,
    when user sends content image.
    """
    if not message.photo:
        await message.answer(messages['content_img_wrong_inp'])
        return

    await state.update_data(content_img_id=message.photo[0].file_id)
    await RequestOrder.next()
    await message.answer(messages['style_img_inp'])
    # await bot.download_file_by_id(img, destination='downloaded_img.jpeg')


@dp.message_handler(state=RequestOrder.style_inp, content_types=ContentType.ANY)
async def get_style_image(message: types.Message,
                          state: FSMContext):
    """
    This handler will be called during inputs collection dialog,
    when user sends style image.
    """
    if not message.photo:
        await message.answer(messages['style_img_wrong_inp'])
        return

    await state.update_data(style_img_id=message.photo[0].file_id)
    await RequestOrder.next()
    await message.answer(messages['gamma_inp'])

@dp.message_handler(state=RequestOrder.gamma_inp, content_types=ContentType.TEXT)
async def set_gamma(message: types.Message,
                    state: FSMContext):
    """
    This handler will be called during inputs collection dialog,
    when user sets gamma parameter.
    """
    if message.text.lower() == 'd':
        await state.update_data(gamma=1.0)
        await RequestOrder.next()
        await message.answer(messages['resize_inp'])
    else:
        try:
            gamma_val = float(message.text)
            if gamma_val > 0 and gamma_val <= 10:
                await state.update_data(gamma=gamma_val)
                await RequestOrder.next()
                await message.answer(messages['resize_inp'])
            else:
                await message.answer(messages['gamma_wrong_inp'])
        except ValueError:
            await message.answer(messages['gamma_wrong_inp'])

@dp.message_handler(state=RequestOrder.resize_inp, content_types=ContentType.TEXT)
async def set_resize(message: types.Message,
                    state: FSMContext):
    """
    This handler will be called during inputs collection dialog,
    when user tells if image has to be resized.
    """
    if message.text.lower() not in {'y', 'n'}:
        await message.answer(messages['resize_wrong_inp'])
        return
    else:
        is_resize = message.text.lower() == 'y'
        await state.update_data(resize=is_resize)
        await state.update_data(user_id=message.from_user.id)
        await message.answer(messages['inputs_received'])

        # get collected input data:
        inputs = await state.get_data()

        # run style transfer:
        await message.answer(messages['processing'])
        await run_nst(inputs)

        # send result to the user:in
        await types.ChatActions.upload_photo()
        media = types.MediaGroup()
        output_img_name = str(inputs['user_id']) + '_output_img.jpg'
        media.attach_photo(types.InputFile(output_img_name), messages['finish'])
        await message.answer_media_group(media=media)
        await state.finish()

@dp.message_handler(content_types=ContentType.ANY, state="*")
async def unspecified_message(message: types.Message):
    """
    This handler will process any unspecified message from the user.
    """
    await message.answer(messages['echo'])


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)