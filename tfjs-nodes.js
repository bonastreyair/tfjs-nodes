module.exports = function(RED) {
    // Common stuff for all nodes
    const tf = require('@tensorflow/tfjs-node')
    const fs = require('fs')
    const path = require('path')
    const express = require('express')
    const compression = require('compression')
    global.fetch = require('node-fetch')

    RED.httpNode.use(compression())
    RED.httpNode.use('/coco', express.static(__dirname + '/models/coco'))
    RED.httpNode.use('/mobilenet', express.static(__dirname + '/models/mobilenet'))
    RED.httpNode.use('/posenet', express.static(__dirname + '/models/posenet'))

    function setNodeStatus(node, status) {
        switch (status) {
            case 'modelReady':
                node.status({ fill: 'green', shape: 'dot', text: 'ready' })
                break
            case 'modelLoading':
                node.status({ fill: 'yellow', shape: 'ring', text: 'loading model...' })
                break
            case 'infering':
                node.status({ fill: 'blue', shape: 'ring', text: 'infering...' })
                break
            case 'modelError':
                node.status({ fill: 'red', shape: 'dot', text: 'model error' })
                break
            case 'error':
                node.status({ fill: 'red', shape: 'dot', text: 'error' })
                break
            case 'close':
                node.status({})
                break
            default:
                node.status({ fill: 'grey', shape: 'dot', text: status })
        }
    }

    async function inputNodeHandler(node, msg, params) {
        try {
            node.success = false
            if (node.ready) {
                let image = msg.payload
                if (typeof image === 'string') { image = fs.readFileSync(image) } // If it's a string assume it's a filename
                const results = await node.inferImage(image, params)
                setNodeStatus(node, 'modelReady')
                node.success = true
                return results
            } else {
                node.error('model is not ready')
            }
        } catch (error) {
            node.error(error)
            setNodeStatus(node, 'error')
        }
    }

    function filterThreshold(results, threshold) {
        for (let i = 0; i < results.length; i++) {
            if (results[i].score < (threshold / 100)) {
                results.splice(i, 1)
                i--
            }
        }
        return results
    }

    function changeKeyResults(results) {
        const out = []
        for (let i = 0; i < results.length; i++) {
            out.push({
                class: results[i].className,
                score: results[i].probability
            })
        }
        return out
    }

    function countClasses(results) {
        const classes = {}
        for (let i = 0; i < results.length; i++) {
            classes[results[i].class] = (classes[results[i].class] || 0) + 1
        }
        return classes
    }

    // Predict Model
    function tensorflowPredict(config) {
        RED.nodes.createNode(this, config)
        this.mode = config.mode
        this.modelUrl = config.modelUrl
        this.localModel = config.localModel
        this.passthru = config.passthru
        this.params = {}

        var node = this

        async function loadModel() {
            setNodeStatus(node, 'modelLoading')
            try {
                node.ready = false
                if (node.mode === 'online') {
                    if (node.modelUrl === '') {
                        setNodeStatus(node, 'set a New URL')
                        return
                    } else {
                        node.model = await tf.loadLayersModel(node.modelUrl)
                        const shape = node.model.inputs[0].shape
                        shape.shift()
                            // node.log('input model shape: ' + shape)
                        shape.unshift(1)
                        node.model.predict(tf.zeros(shape)).dispose()
                        node.shape = shape
                    }
                } else {
                    node.model = await tf.loadLayersModel(node.localModel)
                        //   setNodeStatus(node, 'mode not supported')
                        //   node.ready = false
                        //   return
                        // var url = path.join('.node-red', 'node_modules', 'node-red-contrib-tfjs-nodes')
                        // var modelUrl = 'file://../' + url + '/models/' + node.localModel + '/model.json'
                        // node.model = await mobilenet.load({ modelUrl: modelUrl })
                }
                node.ready = true
                setNodeStatus(node, 'modelReady')
            } catch (error) {
                setNodeStatus(node, 'modelError')
                node.error(error)
            }
        }

        node.inferImage = async function(image, params) {
            setNodeStatus(node, 'infering')
            let tensorImage = tf.node.decodeImage(image, node.shape[3])
                // Rescale the image to fit the wanted shape
            const scaledTensorImage = tf.image.resizeBilinear(tensorImage, [node.shape[1], node.shape[2]], true)
            const offset = tf.scalar(127.5)
                // Normalize the image from [0, 255] to [-1, 1].
            const normalized = scaledTensorImage.sub(offset).div(offset)
            tensorImage = normalized.reshape(node.shape)

            const tensorResult = node.model.predict(tensorImage)
            const indexMax = tensorResult.argMax(1).dataSync()[0]
            const resultsArray = Array.from(tensorResult.dataSync())

            tf.dispose(tensorImage) // Free space

            const results = {
                resultsArray: resultsArray,
                indexMax: indexMax
            }

            return results
        }

        loadModel()

        node.on('input', function(msg) {
            if (node.passthru === true) { msg.image = msg.payload }
            const dynamicParams = {}
            inputNodeHandler(node, msg, dynamicParams).then(
                function(results) {
                    if (node.success) {
                        msg.indexMax = results.indexMax
                        msg.payload = results.resultsArray
                        node.send(msg)
                    }
                })
        })

        node.on('close', function() { setNodeStatus(node, 'close') })
    }

    // MobileNet Model
    function tensorflowMobilenet(config) {
        const mobilenet = require('@tensorflow-models/mobilenet')

        RED.nodes.createNode(this, config)
        this.mode = config.mode
        this.modelUrl = config.modelUrl
        this.passthru = config.passthru
        this.params = {
            threshold: config.threshold
        }

        var node = this

        async function loadModel() {
            setNodeStatus(node, 'modelLoading')
            try {
                node.ready = false
                if (node.modelUrl === '') {
                    node.model = await mobilenet.load()
                } else {
                    node.model = await mobilenet.load({ modelUrl: node.modelUrl, version: 1, alpha: 1.0 })
                }
                node.ready = true
                setNodeStatus(node, 'modelReady')
            } catch (error) {
                setNodeStatus(node, 'modelError')
                node.error(error)
            }
        }

        node.inferImage = async function(image, params) {
            setNodeStatus(node, 'infering')
            const tensorImage = tf.node.decodeImage(image)
            const classification = await node.model.classify(tensorImage)
            const filteredClassification = filterThreshold(changeKeyResults(classification), params.threshold)

            tf.dispose(tensorImage) // Free space

            const results = {
                classification: filteredClassification
            }
            return results
        }

        loadModel()

        node.on('input', function(msg) {
            if (node.passthru === true) { msg.image = msg.payload }
            msg.threshold = parseInt(msg.threshold || node.params.threshold) // Adds to msg if it doesn't exist yet
            const dynamicParams = { threshold: msg.threshold }

            inputNodeHandler(node, msg, dynamicParams).then(
                function(results) {
                    if (node.success) {
                        msg.payload = results.classification
                        node.send(msg)
                    }
                })
        })
        node.on('close', function() { setNodeStatus(node, 'close') })
    }

    // Coco SSD Model
    function tensorflowCocoSsd(config) {
        const cocoSsd = require('@tensorflow-models/coco-ssd')

        RED.nodes.createNode(this, config)
        this.modelUrl = config.modelUrl
        this.passthru = config.passthru
        this.params = {
            threshold: config.threshold,
            maxDetections: config.maxDetections
        }

        var node = this

        async function loadModel() {
            setNodeStatus(node, 'modelLoading')
            try {
                node.ready = false
                if (node.modelUrl === '') {
                    node.model = await cocoSsd.load()
                } else {
                    node.model = await cocoSsd.load({ modelUrl: node.modelUrl })
                }
                node.ready = true
                setNodeStatus(node, 'modelReady')
            } catch (error) {
                setNodeStatus(node, 'modelError')
                node.error(error)
            }
        }

        node.inferImage = async function(image, params) {
            setNodeStatus(node, 'infering')
            const tensorImage = tf.node.decodeImage(image)
            const detections = await node.model.detect(tensorImage, params.maxDetections)
            const filteredDetections = filterThreshold(detections, params.threshold)
            const classes = countClasses(filteredDetections)

            tf.dispose(tensorImage) // Free space

            const results = {
                filteredDetections: filteredDetections,
                classes: classes
            }
            return results
        }

        loadModel()

        node.on('input', function(msg) {
            if (node.passthru === true) { msg.image = msg.payload }
            msg.threshold = parseInt(msg.threshold || node.params.threshold) // Adds to msg if it doesn't exist yet
            msg.maxDetections = parseInt(msg.maxDetections || node.params.maxDetections) // Adds to msg if it doesn't exist yet

            const dynamicParams = { threshold: msg.threshold, maxDetections: msg.maxDetections }

            inputNodeHandler(node, msg, dynamicParams).then(
                function(results) {
                    if (node.success) {
                        msg.payload = results.filteredDetections
                        msg.classes = results.classes
                        node.send(msg)
                    }
                }
            )
        })

        node.on('close', function() { setNodeStatus(node, 'close') })
    }

    // Posenet Model
    function tensorflowPosenet(config) {
        const posenet = require('@tensorflow-models/posenet')

        RED.nodes.createNode(this, config)
        this.modelUrl = config.modelUrl
        this.passthru = config.passthru
        this.params = {
            threshold: config.threshold,
            maxDetections: config.maxDetections
        }

        var node = this

        async function loadModel() {
            setNodeStatus(node, 'modelLoading')
            try {
                node.ready = false
                if (node.modelUrl === '') {
                    node.model = await posenet.load()
                } else {
                    node.model = await posenet.load({ modelUrl: node.modelUrl })
                }
                node.ready = true
                setNodeStatus(node, 'modelReady')
            } catch (error) {
                setNodeStatus(node, 'modelError')
                node.error(error)
            }
        }

        node.inferImage = async function(image, params) {
            setNodeStatus(node, 'infering')
            const tensorImage = tf.node.decodeImage(image)
            const poses = await node.model.estimateMultiplePoses(tensorImage, {
                flipHorizontal: false,
                maxDetections: params.maxDetections,
                scoreThreshold: params.threshold / 100,
                nmsRadius: 20
            })
            const filteredResults = filterThreshold(poses, params.threshold)

            tf.dispose(tensorImage) // Free space

            const results = {
                filteredResults: filteredResults,
                classes: filteredResults.length ? { person: filteredResults.length } : {}
            }
            return results
        }

        loadModel()

        node.on('input', function(msg) {
            if (node.passthru === true) { msg.image = msg.payload }
            msg.threshold = parseInt(msg.threshold || node.params.threshold) // Adds to msg if it doesn't exist yet
            msg.maxDetections = parseInt(msg.maxDetections || node.params.maxDetections) // Adds to msg if it doesn't exist yet

            const dynamicParams = { threshold: msg.threshold, maxDetections: msg.maxDetections }

            inputNodeHandler(node, msg, dynamicParams).then(
                function(results) {
                    if (node.success) {
                        msg.payload = results.filteredResults
                        msg.classes = results.classes
                        node.send(msg)
                    }
                }
            )
        })

        node.on('close', function() { setNodeStatus(node, 'close') })
    }

    // Register Node-RED nodes
    RED.nodes.registerType('tensorflowPredict', tensorflowPredict)
    RED.nodes.registerType('tensorflowMobilenet', tensorflowMobilenet)
    RED.nodes.registerType('tensorflowCocoSsd', tensorflowCocoSsd)
    RED.nodes.registerType('tensorflowPosenet', tensorflowPosenet)
}