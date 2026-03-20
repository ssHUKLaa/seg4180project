#include "CommandServer.h"
#include "PluginManager.h"
#include <objbase.h>
#include <iostream>

namespace
{
struct LoadPluginContext
{
    PluginManager* manager = nullptr;
    juce::File file;
    bool success = false;
    juce::String error;
};

struct EditorCommandContext
{
    PluginManager* manager = nullptr;
    bool show = true;
    bool success = false;
    juce::String error;
};

void* loadPluginOnMessageThread(void* userData)
{
    auto* ctx = static_cast<LoadPluginContext*>(userData);
    if (ctx == nullptr || ctx->manager == nullptr)
        return nullptr;

    ctx->success = ctx->manager->loadPlugin(ctx->file);
    if (!ctx->success)
        ctx->error = ctx->manager->getLastError();
    return nullptr;
}

void* editorCommandOnMessageThread(void* userData)
{
    auto* ctx = static_cast<EditorCommandContext*>(userData);
    if (ctx == nullptr || ctx->manager == nullptr)
        return nullptr;

    if (ctx->show)
    {
        ctx->success = ctx->manager->showEditor();
        if (!ctx->success)
            ctx->error = ctx->manager->getLastError();
    }
    else
    {
        ctx->manager->hideEditor();
        ctx->success = true;
    }
    return nullptr;
}
}

CommandServer::CommandServer(PluginManager& pluginManager)
    : juce::Thread("CommandServer"), pluginMgr(pluginManager)
{
}

CommandServer::~CommandServer()
{
    stop();
}

bool CommandServer::start()
{
    if (isListening)
        return true;

    serverSocket = std::make_unique<juce::StreamingSocket>();
    
    if (!serverSocket->createListener(5057, "127.0.0.1"))
    {
        if (onStatusChange)
            onStatusChange("Failed to bind to port 5057");
        return false;
    }

    isListening = true;
    startThread(juce::Thread::Priority::normal);
    
    if (onStatusChange)
        onStatusChange("Server listening on 127.0.0.1:5057");
    
    return true;
}

void CommandServer::stop()
{
    isListening = false;
    
    if (serverSocket)
        serverSocket->close();
    
    if (clientSocket)
        clientSocket->close();
    
    stopThread(5000);
}

void CommandServer::run()
{
    HRESULT comHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool shouldUninitCom = (comHr == S_OK || comHr == S_FALSE);

    while (isListening && !threadShouldExit())
    {
        if (!serverSocket)
            break;

        // Accept incoming connection (blocking, timeout 1000ms)
        auto incomingSocket = serverSocket->waitForNextConnection();
        
        if (!incomingSocket)
            continue;

        clientSocket = std::unique_ptr<juce::StreamingSocket>(incomingSocket);

        // Read JSON command
        juce::MemoryOutputStream buffer;
        char readBuf[4096];
        int bytesRead = 0;

        while ((bytesRead = clientSocket->read(readBuf, sizeof(readBuf), true)) > 0)
        {
            buffer.write(readBuf, bytesRead);
            
            // Simple check: if we got a complete JSON object, process it
            auto str = buffer.toString();
            if (str.contains("}"))
                break;
        }

        auto jsonStr = buffer.toString();
        
        if (jsonStr.isEmpty())
        {
            clientSocket->close();
            continue;
        }

        try
        {
            // Parse and handle command
            juce::var jsonCmd = juce::JSON::parse(jsonStr);
            juce::var response;

            if (!jsonCmd.isObject())
            {
                auto obj = new juce::DynamicObject();
                response = juce::var(obj);
                response.getDynamicObject()->setProperty("ok", false);
                response.getDynamicObject()->setProperty("error", "Invalid JSON");
            }
            else
            {
                handleCommand(jsonCmd, response);
            }

            // Send response
            auto responseStr = juce::JSON::toString(response);
            clientSocket->write(responseStr.toRawUTF8(), (int) responseStr.getNumBytesAsUTF8());
            clientSocket->close();
        }
        catch (const std::exception& e)
        {
            auto obj = new juce::DynamicObject();
            juce::var response(obj);
            response.getDynamicObject()->setProperty("ok", false);
            response.getDynamicObject()->setProperty("error", juce::String("Host exception: ") + e.what());
            auto responseStr = juce::JSON::toString(response);
            clientSocket->write(responseStr.toRawUTF8(), (int) responseStr.getNumBytesAsUTF8());
            clientSocket->close();
        }
        catch (...)
        {
            auto obj = new juce::DynamicObject();
            juce::var response(obj);
            response.getDynamicObject()->setProperty("ok", false);
            response.getDynamicObject()->setProperty("error", "Host exception: unknown");
            auto responseStr = juce::JSON::toString(response);
            clientSocket->write(responseStr.toRawUTF8(), (int) responseStr.getNumBytesAsUTF8());
            clientSocket->close();
        }
    }

    if (shouldUninitCom)
        CoUninitialize();
}

void CommandServer::handleCommand(const juce::var& jsonCmd, juce::var& response)
{
    auto obj = new juce::DynamicObject();
    response = juce::var(obj);
    
    auto cmd = jsonCmd.getProperty("cmd", "").toString();
    auto id = jsonCmd.getProperty("id", "unnamed").toString();

    response.getDynamicObject()->setProperty("id", id);

    if (cmd == "load_plugin")
    {
        auto path = jsonCmd.getProperty("path", "").toString();
        handleLoadPlugin(path, response);
    }
    else if (cmd == "load_midi")
    {
        auto path = jsonCmd.getProperty("path", "").toString();
        handleLoadMidi(path, response);
    }
    else if (cmd == "play")
    {
        handlePlay(response);
    }
    else if (cmd == "stop")
    {
        handleStop(response);
    }
    else if (cmd == "get_status")
    {
        handleGetStatus(response);
    }
    else if (cmd == "set_parameter")
    {
        int paramIndex = int(jsonCmd.getProperty("index", -1));
        float value = float(jsonCmd.getProperty("value", 0.0));
        handleSetParameter(paramIndex, value, response);
    }
    else if (cmd == "show_editor")
    {
        handleShowEditor(response);
    }
    else if (cmd == "hide_editor")
    {
        handleHideEditor(response);
    }
    else
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "Unknown command: " + cmd);
    }
}

void CommandServer::handleLoadPlugin(const juce::String& path, juce::var& response)
{
    std::cout << "CommandServer: load_plugin requested: " << path << std::endl;
    std::cout.flush();
    juce::Logger::writeToLog("CommandServer: load_plugin requested: " + path);
    if (path.isEmpty())
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "Plugin path required");
        return;
    }

    juce::File pluginFile(path);
    if (!pluginFile.existsAsFile() && !pluginFile.isDirectory())
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "Plugin file not found");
        return;
    }

    auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    if (mm == nullptr)
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "MessageManager not initialized");
        return;
    }

    LoadPluginContext ctx;
    ctx.manager = &pluginMgr;
    ctx.file = pluginFile;
    mm->callFunctionOnMessageThread(loadPluginOnMessageThread, &ctx);
    const bool success = ctx.success;

    response.getDynamicObject()->setProperty("ok", success);
    if (success)
    {
        response.getDynamicObject()->setProperty("name", pluginMgr.getPluginName());
    }
    else
    {
        response.getDynamicObject()->setProperty("error", ctx.error.isNotEmpty() ? ctx.error : pluginMgr.getLastError());
    }
}

void CommandServer::handleLoadMidi(const juce::String& path, juce::var& response)
{
    if (path.isEmpty())
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "MIDI path required");
        return;
    }

    juce::File midiFile(path);
    if (!midiFile.exists())
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "MIDI file not found");
        return;
    }

    const bool success = pluginMgr.loadMidiFile(midiFile);
    response.getDynamicObject()->setProperty("ok", success);
    if (success)
        response.getDynamicObject()->setProperty("message", "MIDI loaded: " + path);
    else
        response.getDynamicObject()->setProperty("error", pluginMgr.getLastError());
}

void CommandServer::handlePlay(juce::var& response)
{
    if (!pluginMgr.isPluginLoaded())
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "No plugin loaded");
        return;
    }

    const bool success = pluginMgr.startPlayback();
    response.getDynamicObject()->setProperty("ok", success);
    if (success)
        response.getDynamicObject()->setProperty("message", "Playback started");
    else
        response.getDynamicObject()->setProperty("error", pluginMgr.getLastError());
}

void CommandServer::handleStop(juce::var& response)
{
    pluginMgr.stopPlayback();
    response.getDynamicObject()->setProperty("ok", true);
    response.getDynamicObject()->setProperty("message", "Playback stopped");
}

void CommandServer::handleGetStatus(juce::var& response)
{
    response.getDynamicObject()->setProperty("ok", true);
    response.getDynamicObject()->setProperty("plugin_loaded", pluginMgr.isPluginLoaded());
    response.getDynamicObject()->setProperty("plugin_name", pluginMgr.getPluginName());
    response.getDynamicObject()->setProperty("midi_loaded", pluginMgr.isMidiLoaded());
    response.getDynamicObject()->setProperty("is_playing", pluginMgr.isPlaying());
}

void CommandServer::handleSetParameter(int paramIndex, float value, juce::var& response)
{
    if (paramIndex < 0)
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "Invalid parameter index");
        return;
    }

    pluginMgr.setParameter(paramIndex, value);
    response.getDynamicObject()->setProperty("ok", true);
    response.getDynamicObject()->setProperty("message", "Parameter set");
}

void CommandServer::handleShowEditor(juce::var& response)
{
    auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    if (mm == nullptr)
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "MessageManager not initialized");
        return;
    }

    EditorCommandContext ctx;
    ctx.manager = &pluginMgr;
    ctx.show = true;
    mm->callFunctionOnMessageThread(editorCommandOnMessageThread, &ctx);

    response.getDynamicObject()->setProperty("ok", ctx.success);
    if (ctx.success)
        response.getDynamicObject()->setProperty("message", "Editor opened");
    else
        response.getDynamicObject()->setProperty("error", ctx.error.isNotEmpty() ? ctx.error : pluginMgr.getLastError());
}

void CommandServer::handleHideEditor(juce::var& response)
{
    auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    if (mm == nullptr)
    {
        response.getDynamicObject()->setProperty("ok", false);
        response.getDynamicObject()->setProperty("error", "MessageManager not initialized");
        return;
    }

    EditorCommandContext ctx;
    ctx.manager = &pluginMgr;
    ctx.show = false;
    mm->callFunctionOnMessageThread(editorCommandOnMessageThread, &ctx);

    response.getDynamicObject()->setProperty("ok", true);
    response.getDynamicObject()->setProperty("message", "Editor closed");
}
