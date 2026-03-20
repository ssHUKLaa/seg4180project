#pragma once

#include <JuceHeader.h>
#include <memory>
#include <functional>

class PluginManager;

class CommandServer : private juce::Thread
{
public:
    CommandServer(PluginManager& pluginManager);
    ~CommandServer();

    bool start();
    void stop();
    bool isRunning() const { return isListening; }

    std::function<void(const juce::String&)> onStatusChange;

private:
    void run() override;
    
    PluginManager& pluginMgr;
    std::unique_ptr<juce::StreamingSocket> serverSocket;
    std::unique_ptr<juce::StreamingSocket> clientSocket;
    bool isListening = false;

    void handleCommand(const juce::var& jsonCmd, juce::var& response);
    void handleLoadPlugin(const juce::String& path, juce::var& response);
    void handleLoadMidi(const juce::String& path, juce::var& response);
    void handlePlay(juce::var& response);
    void handleStop(juce::var& response);
    void handleGetStatus(juce::var& response);
    void handleSetParameter(int paramIndex, float value, juce::var& response);
    void handleShowEditor(juce::var& response);
    void handleHideEditor(juce::var& response);

};
