import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart, StateFilter

from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import StatesGroup, State, default_state
from aiogram.fsm.context import FSMContext
from aiogram.types import (FSInputFile, CallbackQuery, InlineKeyboardButton,
                           InlineKeyboardMarkup, Message)

import os
import matplotlib.pyplot as plt

from model import Model
from models import Models, ResidualBlock, GeneratorResNet

logging.basicConfig(level=logging.INFO)

models_path = os.environ.get('MODELS_PATH')
model_path = os.path.join(models_path, 'vgg.pk')
token = os.environ.get('BOT_TOKEN')
print(token)
bot = Bot(token=token)
storage = MemoryStorage()

dp = Dispatcher(storage=storage)

user_dict: dict[int, dict[str, str | int | bool]] = {}


class FSMFillForm(StatesGroup):
	style_path = State()
	content_path = State()
	yes_no = State()
	photo_mode = State()
	photo_path = State()

@dp.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message):
    await message.answer(
        'Привет!\nДавай сгенерируем новое фото!\n\n'
        'Чтобы узнать как - отправьте команду /help'
    )

@dp.message(Command(commands='cancel'), StateFilter(default_state))
async def process_cancel_command(message: Message):
    await message.answer(
        text='Отменять нечего.\n\n'
             'отправьте команду /help'
    )

@dp.message(Command(commands='cancel'), ~StateFilter(default_state))
async def process_cancel_command_state(message: Message, state: FSMContext):
    await message.answer(
        text='Чтобы начать заново - \n'
             'отправьте команду /transfer'
    )
    await state.clear()

@dp.message(Command(commands='help'), StateFilter(default_state))
async def process_help_command(message: Message):
    await message.answer(
		'Отправьте команду /transfer,\n'
		'чтобы начать style transfer\n\n'
		'Отправьте команду /style,\n'
		'чтобы стилизировать фото в\n один из предложенных стилей'
    )

@dp.message(Command(commands='transfer'), StateFilter(default_state))
async def process_transfer_command(message: Message, state: FSMContext):
	await message.answer('Пришлите фото нового стиля')
	await state.set_state(FSMFillForm.style_path)

@dp.message(StateFilter(FSMFillForm.style_path), F.photo)
async def process_style_sent(message: Message, state: FSMContext):
	style_path = f"/tmp/{message.photo[-1].file_id}.jpg"
	await bot.download(
			message.photo[-1],
			destination=style_path
		)
	await state.update_data(style_path=style_path)
	await message.answer('Хорошо, теперь пришлите фото,\n' 
	'которое Вы хотите изменить')
	await state.set_state(FSMFillForm.content_path)

@dp.message(StateFilter(FSMFillForm.style_path))
async def warning_not_style(message: Message):
    await message.answer(
        text='Это не похоже на фото стиля.'
			'Если вы хотите начать заново '
            '- отправьте команду /cancel'
    )

@dp.message(StateFilter(FSMFillForm.content_path), F.photo)
async def process_content_sent(message: Message, state: FSMContext):
	content_path = f"/tmp/{message.photo[-1].file_id}.jpg"
	await bot.download(
			message.photo[-1],
			destination=content_path
		)
	await state.update_data(content_path=content_path)
	user_dict[message.from_user.id] = await state.get_data()
	await message.answer('Отлично.\n\n'
						'Начинаем генерацию? (Да \ Нет)')
	await state.set_state(FSMFillForm.yes_no)
	
@dp.message(StateFilter(FSMFillForm.content_path))
async def warning_not_content(message: Message):
    await message.answer(
        text='Это не похоже на фото.'
			'Если вы хотите начать заново '
            '- отправьте команду /cancel'
    )

@dp.message(StateFilter(FSMFillForm.yes_no), F.text.in_(['Да']))
async def style_photo(message: Message, state: FSMContext):
	if message.from_user.id in user_dict:
		await message.answer(
            text='Пожалуйста, пожождите.'
			'Это займет пару минут'
        )
		style_path = user_dict[message.from_user.id]['style_path']
		content_path = user_dict[message.from_user.id]['content_path']
		model = Model(style_path, content_path, model_path)
		photo = model.vgg()

		photo_path = f"/tmp/generated{message.from_user.id}.jpg"
		plt.imsave(photo_path, photo)
		
		photo = FSInputFile(photo_path)
		await message.answer_photo(photo)
		user_dict.pop(message.from_user.id)
		for path in [style_path, content_path, photo_path]:
			os.remove(path)
		await state.clear()

	else:
		await message.answer(
            text='Вы еще не отправили фото. Чтобы начать,'
            'отправьте команду /transfer'
        )

@dp.message(StateFilter(FSMFillForm.yes_no))
async def warning_not_yes_no(message: Message):
	await message.answer(
        text='Ответ должен был быть Да!'
    )

@dp.message(Command(commands='style'), StateFilter(default_state))
async def process_style_command(message: Message, state: FSMContext):
	await message.answer('Выберите стиль')
	monet_button = InlineKeyboardButton(
        text='Клод Моне',
        callback_data='monet'
    )
	cezanne_button = InlineKeyboardButton(
        text='Поль Сезанн',
        callback_data='cezanne'
    )
	vangogh_button = InlineKeyboardButton(
        text='Винсент ван Гог',
        callback_data='vangogh'
    )
	ukiyoe_button = InlineKeyboardButton(
        text='Укиё-э',
        callback_data='ukiyoe'
    )
	keyboard: list[list[InlineKeyboardButton]] = [
        [monet_button, cezanne_button],
        [vangogh_button, ukiyoe_button]
    ]
	markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
	await message.answer(
        text='А теперь отправьте фото, которое будем стилизовать)',
        reply_markup=markup
    )
	await state.set_state(FSMFillForm.photo_mode)

@dp.callback_query(StateFilter(FSMFillForm.photo_mode),
                   F.data.in_(['monet', 'cezanne', 'vangogh', 'ukiyoe']))
async def process_style_press(callback: CallbackQuery, state: FSMContext):
	await state.update_data(photo_mode=callback.data)
	await callback.message.delete()
	await callback.message.answer(
        text='А теперь пришлите фото, которое будем стилизовать'
    )
	await state.set_state(FSMFillForm.photo_path)

@dp.message(StateFilter(FSMFillForm.photo_path), F.photo)
async def process_photo_sent(message: Message, state: FSMContext):
	photo_path = f"/tmp/{message.photo[-1].file_id}.jpg"
	await bot.download(
			message.photo[-1],
			destination=photo_path
		)
	await state.update_data(photo_path=photo_path)
	await message.answer(
            text='Пожалуйста, пожождите.'
			'Это займет пару секунд'
        )
	user_dict[message.from_user.id] = await state.get_data()
	photo_path = user_dict[message.from_user.id]['photo_path']
	photo_mode = user_dict[message.from_user.id]['photo_mode']
	path_to_model = os.path.join(models_path, f'{photo_mode}.onnx')
	model = Models(path_to_model, photo_path)
	photo = model.run()

	generated_path = f"/tmp/generated{message.from_user.id}.jpg"
	plt.imsave(generated_path, photo)
	
	photo = FSInputFile(generated_path)
	await message.answer_photo(photo)
	user_dict.pop(message.from_user.id)
	for path in [generated_path, photo_path]:
		os.remove(path)
	await state.clear()


@dp.message(StateFilter(FSMFillForm.photo_path))
async def warning_not_photo(message: Message):
    await message.answer(
        text='Это не похоже на фото стиля.'
			'Если вы хотите начать заново '
            '- отправьте команду /cancel'
    )


@dp.message(StateFilter(default_state))
async def send_echo(message: Message):
    await message.reply(text='Извините, моя твоя не понимать')

if __name__ == '__main__':
	dp.run_polling(bot)
