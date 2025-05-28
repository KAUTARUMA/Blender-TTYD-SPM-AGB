# Blender TTYD/SPM AGB Importer/Exporter
## WARNING: This tool is only partly functional, multiple things have not been implemented or tested. Contributions are always appreciated!

Notable unimplemented/untested features are
1. Importing Animations
2. Importing/Exporting Texture/UV Animations
3. An accessible loop toggle (All animations are looping by default)

Each animation must have at least 2 keyframes containing an object.
Currently, groups are imported as empties, meaning you have to create an action and add it to the NLA for each object. This will be replaced with bones at a later date.

A proper tutorial will come eventually, but for now just look at [this video](https://discord.com/channels/480157509261459468/846130782065131521/1377098630321668289). ((if you cant see it join [this server](https://discord.gg/pgUvzTE5E5)))

Huge credit goes to Diagamma for their original AGB importer. https://git.gauf.re/antoine/ttyd-stuff/src/branch/master/blender_io_ttyd_agb
I am merely picking up where they left off.
