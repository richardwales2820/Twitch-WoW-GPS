# WoW-CLASSifier
Deep net classifier to view a twitch stream and infer the class the player is playing as.

## Training data
Use the class trial feature to create a near-max level chracter of each class. Change to each class specialization and take screenshots of every spell icon as it is idle in the action bar, as well as its state when under cooldown every frame tick. Also, artificially inflate the number of training images by placing fake WoW cursors over various parts of the image to improve detection when obscured by the mouse.