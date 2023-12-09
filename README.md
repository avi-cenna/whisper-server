# whisper-server
Whisper transcription server using ZeroMQ and faster-whisper. Inspired by [WhisperWriter](https://github.com/savbell/whisper-writer).

## Installation
```sh
git clone https://github.com/avi-cenna/whisper-server
cd whisper-server
pyenv install 3.11
pipx install --editable --python=$(pyenv shell 3.11; pyenv which python) .
```

## Usage
whisper-server is meant to be called by other programs, such as Hammerspoon. Here is an example of how to use it with Hammerspoon.
This will bind the hotkey `alt + 1` to call whisper-server, which will then send the keystrokes to the active window.

There are a couple important points to note:
1. The [yourName] in the path to whisper-server must be replaced with your username.
2. The whisper-server must be started before the hotkey is pressed. This can be done by running `whisper-server serve` in a terminal.

```lua
function createCanvas()
    local mousePoint = hs.mouse.getAbsolutePosition()
    local canvasSize = hs.geometry.size(200, 100)
    local canvasFrame = hs.geometry.rect(mousePoint.x, mousePoint.y, canvasSize.w, canvasSize.h)

    local myCanvas = hs.canvas.new(canvasFrame):appendElements({
        type = "rectangle",
        action = "fill",
        fillColor = { red = 0.5, green = 0.5, blue = 0.5, alpha = 0.8 },
        roundedRectRadii = { xRadius = 10, yRadius = 10 },
    }, {
        type = "text",
        text = "Recording...",
        textColor = { red = 1, green = 1, blue = 1, alpha = 1 },
        textSize = 15,
        frame = { x = "10%", y = "10%", w = "80%", h = "80%" },
    })

    return myCanvas
end

local function zmqClientStart()
    local command = "/Users/[yourName]/.local/bin/whisper-server"
    local arguments = { 'send' }
    local canvas = createCanvas()

    -- Create and configure the task
    local task = hs.task.new(command, function(exitCode, stdOut, stdErr)
        print(stdOut)
        print(stdErr)
        if exitCode == 0 then
            notificationInfo = stdOut
        else
            notificationInfo = stdErr
        end

        -- send multiple keys
        hs.eventtap.keyStrokes(stdOut)
        canvas:hide()
    end, arguments)

    -- Start the task
    task:start()
    sleep(0.100) -- sleep for 100ms
    canvas:show()
end

-- Bind a hotkey to the zmqClientStart function
hs.hotkey.bind("alt", "1", zmqClientStart)
```

## Configuration
The default configuration file is located at `resources/default_config.yaml`. This file can be overridden
by creating a file in the same directory called `user_config.yaml`.

## TODO
 - Document configuration options