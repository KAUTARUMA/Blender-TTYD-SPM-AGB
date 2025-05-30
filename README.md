# Blender TTYD/SPM AGB Importer/Exporter
## WARNING: This tool is only partly functional, multiple things have not been implemented or tested. Contributions are always appreciated!

Notable unimplemented/untested features are
1. Importing/Exporting Texture/UV Animations

Notable bugs include
1. Animations import/export slightly incorrectly, i will be slowly chipping away at fixing this
2. Material settings are incorrect in game.

Each animation must have at least 2 keyframes.
The seperate `_vis` tracks are for object visibility, this is controlled by the "Show In Viewport" option (you have to enable it in the project tree)

A proper tutorial will come eventually, but for now just look at [this video](https://discord.com/channels/480157509261459468/846130782065131521/1377098630321668289). ((if you cant see it join [this server](https://discord.gg/pgUvzTE5E5)))

Huge credit goes to Diagamma for their original [AGB importer](https://git.gauf.re/antoine/ttyd-stuff/src/branch/master/blender_io_ttyd_agb), I am merely picking up where they left off.
