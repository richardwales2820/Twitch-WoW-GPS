# Twitch-WoW-GPS
Runs CV against top streams and returns the zone/dungeon the player is currently in

## Two Approaches
### OCR
By cropping the stream's frame to just the location string above the mini-map, OCR can be done on the image and the location can be searched to return a probable zone or instance that the streamer is currently playing in. OCR net would need to be trained on the WoW font. Also, many top streamers do not use the default WoW UI. This might need to be dynamic to FIND the mini-map and location string, if it even exists on their UI.

### Zone CV
Test data of thousands of images can be gathered of a certain zone and be labeled with their location. By training a deep net with these images and locations, a stream could then be analyzed and attempt to place them in a certain zone/instance. Initially we could start small with just the new BFA zones and dungeons. Could be expanded if interest is there.

Test data should be obtained manually as Blizzard has ToS rules against bots.