import re


class Translation:
    def __init__(self):
        # 2-digit language code
        self.id = 'en'
        # 3-digit language code
        self.code = 'eng'
        # Unicode flag
        self.flags = ['🇺🇸']
        # Supported by OCR Engine 2
        self.supported = True

        self.SERVER_URL = 'https://discord.gg/SyGmBxds3M'
        self.BOT_URL = 'https://discord.com/api/oauth2/authorize?client_id=878242722429931551&permissions=19456&scope=bot'
        self.GITHUB_URL = 'https://github.com/hrfmsd/Genshin-Status-Rater'
        self.SAMPLE_URL = 'https://cdn.discordapp.com/attachments/787533173004173343/790751503475802122/unknown.png'

        # stats as they appear in-game
        self.hp = 'HP'
        self.hp_max = 'HP Max'
        self.heal = 'Healing'
        self.df = 'DEF'
        self.er = 'Energy Recharge'
        self.em = 'Elemental Mastery'
        self.atk = 'ATK'
        self.atk_base = 'ATK Base'
        self.atk_add = 'ATK Add'
        self.cd = 'CRIT DMG'
        self.cr = 'CRIT Rate'
        self.phys = 'Physical DMG'
        self.elem = 'Elemental DMG'
        self.anemo = 'Anemo DMG'
        self.elec = 'Electro DMG'
        self.pyro = 'Pyro DMG'
        self.hydro = 'Hydro DMG'
        self.cryo = 'Cryo DMG'
        self.geo = 'Geo DMG'
        self.dend = 'Dendro DMG'

        # text that appears below artifact stats (2-piece set)
        self.piece_set = 'Piece Set'

        # lines will be ignored if they're an exact match
        self.ignore = ['in']
        self.ignore_regs = []
        self.replace = {}

        # text for bot messages
        self.title = 'Status Rating'
        self.lvl = 'Level'
        self.score = 'Gear Score'
        self.score_suffix = ''
        self.exp_dmg = 'Expected Damage'
        self.ideal_dmg_diff = 'DMG Diff'
        self.current_val = '現在値'
        self.ideal_val = '理想値'
        self.ideal_dmg_diff = 'DMG Diff'
        self.join = f'For issues, join the [Artifact Rater Server]({self.SERVER_URL})'
        self.feedback = f'Feedback received, please join {self.SERVER_URL} if you\'d like to add more details'
        self.deprecated = 'Deprecated, please use the `-user lang <lang>` command to set your language'
        self.set_lang = 'Language set to English'
        self.set_prefix = 'Prefix set to %s'
        self.del_preset = 'Preset %s deleted'
        self.set_preset = 'Preset %s set to %s'
        self.no_presets = 'No presets found'

        # text for bot errors
        self.err = 'Error'
        self.err_not_found = 'Error: No image or url found, please make sure they were sent in the same message'
        self.err_parse = 'Error: Command cannot be parsed, please double check the format and spelling'
        self.err_try_again = 'please try again in a few minutes'
        self.err_unknown_ocr = 'Error: OCR failed with unknown error'
        self.err_unknown = 'Unknown error, make sure your language is set (see `-help`) and try using an image from the inventory\'s artifact page'
        self.err_admin_only = 'Error: Only server admins can perform this action'
        self.err_server_only = 'Error: This action can only be performed on servers'

        # help text
        self.help_stats = '`stat` can be one of `hp`, `hp%`, `def`, `def%`, `atk`, `atk%`, `er` (Energy Recharge), `em` (Elemental Mastery), `phys` (Physical DMG), `elem` (Elemental DMG), `cr` (Crit Rate), `cd` (Crit Damage), `heal` (Healing Bonus).'

        self.help_commands = {
            'rate': [
                '-rate <image/url> [preset] [lvl=<level>] [weights]',
                f'''
				Rate an artifact against an optimal 5* artifact. Put the command and image in the same message. Try to use a clear screenshot for the best results.
				If you are on Windows 10, you can use Shift + Windows + S, drag your cursor over the artifact stats and then paste it on discord with Ctrl + V.
				This bot will use default weights (see below) unless you specify your own or select a preset. You can also specify the level you want to compare your artifact to.

				**Default weights**
				ATK%, DMG%, Crit - 1
				ATK, EM, Recharge – 0.5
				Everything else - 0

				**Parameters**
				`image/url`
				The image to be rated, either attached as a file or by putting the url in the message. [Sample]({self.SAMPLE_URL})

				`preset`
				The preset selection of weights to use. See `-presets` for which presets are available, or `-help` for how to set your own.

				`lvl`
				The level of the artifact to compare against, from 0 to 20. Sometimes the auto-detection for level is wrong, use this to correct it.

				`weights`
				The weights to use for rating this artifact. Each weight is of the format `<stat>=<value>`, where `value` is a number between 0 and 1.
				{self.help_stats}

				**Examples**
				`-rate <image> atk%=0 hp=1 er=0.5`
				`-rate <url> support lvl=4`
				'''
            ],

            'feedback': [
                '-feedback <message> [image]',
                'Send direct feedback with up to one image. Use this to send ideas or report errors to help us improve the bot.'
            ],

            'sets': [
                '-sets',
                '''
                View all available presets. Includes personal, server, and default presets.
                This command will display a list containing the name of the preset, where it's from, and the weights it has set.
                '''
            ],

            'lang': [
                '-[user/server] lang <lang>',
                '''
                Set your language for all commands to the 2 letter language code `lang`.
                Artifact Rater will use this language for the images you send in the `-rate` command.

                Languages: English (en), Spanish (es), German (de), French (fr), Portuguese (pt), Polish (pl), Italian (it), Russian (ru), Indonesian (id), Vietnamese (vi), Japanese (ja), Traditional Chinese (tw), Simplified Chinese (cn)
                '''
            ],

            'prefix': [
                '-server prefix <prefix>',
                'Change the bot\'s prefix for this server.'
            ],

            'preset': [
                '-[user/server] preset <name> <weights>',
                f'''
				Create a preset called `name` to use when rating artifacts.
				If you want to check multiple artifacts with the same set of weights, you can use this command to create a preset with the desired weights.
				`weights` will be used in the `-rate` command when the preset is used. `weights` should be in the format `<stat>=<value>`, where `value` is a number between 0 and 1.
				{self.help_stats}

				**Example**
				`-user preset healer hp=0.5 hp%=1 atk%=0`
				`-rate <image> healer`

				`-[user/server] preset delete <names>`

				Delete the presets in `names` (separated by spaces).
				'''
            ]
        }

        self.help_title = 'Status Rater Help'

        self.help_description = f'''
		**Commands**

		`{self.help_commands['rate'][0]}`
		Rate your artifact by sending an image of it. See `-help rate` for more details.

		`{self.help_commands['feedback'][0]}`
		{self.help_commands['feedback'][1]}

		`{self.help_commands['sets'][0]}`
		View all available presets.

		`-help <command>`
		Show the help message for that command. Commands: {', '.join([f'`{command}`' for command in self.help_commands])}.

		**Config**

		`-user` changes your personal config. Overrides server default settings.
		`-server` admin-only, changes the server default.

		`{self.help_commands['prefix'][0]}`
		{self.help_commands['prefix'][1]}

		`{self.help_commands['lang'][0]}`
		Set your language for all commands to the 2 letter language code `lang`. You can also use the flag reactions to change languages.

		`{self.help_commands['preset'][0]}`
		Create a preset to be used when rating artifacts. `weights` will be used in the `-rate` command when the preset is used.

		`-[user/server] preset delete <names>`
		Delete presets.
		'''

        self.source = 'Source Code'
        self.invite = 'Bot Invite'
        self.support = 'Support'
        self.github = f'[GitHub]({self.GITHUB_URL})'
        self.discord = f'[Link]({self.BOT_URL})'
        self.server = f'[Discord]({self.SERVER_URL})'

        self.help_footer = 'To change languages click on the corresponding flag below'


class en(Translation):
    pass


class ja(Translation):
    def __init__(self):
        super().__init__()

        self.id = 'ja'
        self.code = 'jpn'
        self.flags = ['🇯🇵']
        self.supported = False

        self.hp = 'HP'
        self.hp_max = 'HP上限'
        self.heal = '治療効果'
        self.df = '防御力'
        self.er = '元素チャージ効率'
        self.em = '元素熟知'
        self.em_effect_1 = '増幅系'
        self.em_effect_2 = '転化系'
        self.em_effect_3 = '吸収量'
        self.atk = '攻撃力'
        self.atk_base = '基礎攻撃力'
        self.atk_add = '加算攻撃力'
        self.atk_add_rate = '加算攻撃力％'
        self.cd = '会心ダメージ'
        self.cr = '会心率'
        self.phys = '物理ダメージ'
        self.elem = '元素ダメージ'
        self.anemo = '風元素ダメージ'
        self.elec = '雷元素ダメージ'
        self.pyro = '炎元素ダメージ'
        self.hydro = '水元素ダメージ'
        self.cryo = '氷元素ダメージ'
        self.geo = '岩元素ダメージ'
        self.dend = '草元素ダメージ'

        self.piece_set = '2セット'

        self.ignore = []
        self.ignore_regs = [
            '基本ステータス',
            '元素熟知が高いほど、強力な元素の力を発動できる。',
            '蒸発、溶解反応によるダメージ',
            '過負荷、超電導、感電、氷砕き、拡散反応によるダメージ',
            '結晶反応が結晶シールドを生成し、ダメージ吸収量',
            '高級ステータス',
        ]
        self.replace = {
            'カ': '力',
            '①': '',
            '◆': '',
            'X': '',
            '+1,3行': '+1,377',
        }

        self.title = 'ステータス診断'
        self.lvl = 'レベル'
        self.score = 'スコア'
        self.score_suffix = '点'
        self.exp_dmg = 'ダメージ期待値（攻撃力×会心）'
        self.ideal_dmg_diff = '理想値とのダメージ差'
        self.join = f'[公式サーバー]({self.SERVER_URL})に参加する'
        self.feedback = f'フィードバックを受け取りました。詳細を追加したい場合は、 ({self.SERVER_URL})に参加してください。'
        self.set_lang = 'Language set to Japanese'

        self.err = 'エラー'
        self.err_not_found = 'エラー：画像またはURLが見つかりませんでした。同じメッセージで送信されたことを確認してください。'
        self.err_parse = 'エラー：コマンドを解析できません。形式とスペルを再確認してください。'
        self.err_try_again = 'エラー：数分後にもう一度お試しください。'
        self.err_unknown_ocr = 'エラー：OCRが不明なエラーで失敗しました。'
        self.err_unknown = '不明なエラーが発生しました。画像を確認してください。'

        self.help_stats = '`stat`において`hp`、`hp%`、`def`、`def%`(防御力)、`atk`、`atk%`(攻撃力)、`er`(元素チャージ効率)、`em`(元素熟知)、`phys`(物理ダメージ)、`elem`(元素ダメージ)、`cr`(会心率)、`cd`(会心ダメージ)、`heal`(治癒効果)を使えることができます。'

        self.help_description = f'''
		`-rate <image/url> [lvl=<レベル>] [<stat>=<デフォルトの重み付け> ...]`

		{self.help_stats}
		'''


languages = {lang.id: lang for lang in [en(), ja()]}
