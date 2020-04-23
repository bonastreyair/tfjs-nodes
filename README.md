# node-red-contrib-tjfs-nodes
[![Platform](https://img.shields.io/badge/platform-Node--RED-red)](https://nodered.org)
<br>[![JavaScript Style Guide](https://img.shields.io/badge/code_style-standard-brightgreen.svg)](https://standardjs.com)
[![GitHub license](https://img.shields.io/github/license/dceejay/tfjs-nodes)](https://github.com/dceejay/tfjs-nodes/blob/master/LICENSE)

A `Node-RED` node for `tensorflow.js`. A simple function like node preloaded with tensorflowjs and some other useful libraries.

## Install
You have two options to install the node.
 * Use `Manage palette` option in `Node-RED` Menu
 * Run the following command in your Node-RED user directory - typically `~/.node-red`
 ```
 npm install node-contrib-tfjs-nodes
 ```
**Note:** You need to restart `Node-RED` after installation. If installation goes wrong please open a [new issue](https://github.com/dceejay/tfjs-nodes/issues/new/choose).

## Overview
This package contains 4 nodes
 - **tf predict** - Runs tf.predict against any tensorflow model you import
 - **tf mobilenet** - Runs the pretrained mobilenet classifier
 - **tf coco ssd** - Runs the COCO Single Shot Detector against a file or a buffer of a jpg image
 - **tf posenet** - Runs the Posenet body parts pretrained model against a file or a buffer of a jpg image

**Note:** `tensorflow.js` is only available in 64-bit, so it will not run on 32-bit operating systems like Raspbian.
