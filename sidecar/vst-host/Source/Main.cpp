/*
  ==============================================================================

    Headless VST3 Host - JSON command server
    Listens on localhost:5057 for MIDI control and plugin operations

  ==============================================================================
*/

#include <JuceHeader.h>
#include "PluginManager.h"
#include "CommandServer.h"
#include <iostream>
#include <objbase.h>

//==============================================================================
int main(int argc, char* argv[])
{
  juce::ignoreUnused(argc, argv);

    // Output diagnostics to console/stdout
    std::cout << "VST3 Host starting..." << std::endl;

  const HRESULT comHr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  const bool shouldUninitCom = (comHr == S_OK || comHr == S_FALSE);
    
    juce::initialiseJuce_GUI();
    std::cout << "JUCE initialized" << std::endl;

    // Create plugin manager
    auto pluginMgr = std::make_unique<PluginManager>();

    // Create command server
    auto server = std::make_unique<CommandServer>(*pluginMgr);

    // Set up logging
    server->onStatusChange = [](const juce::String& status)
    {
        std::cout << "[Server] " << status << std::endl;
        juce::Logger::writeToLog("[Server] " + status);
    };

    // Start server
    std::cout << "Starting command server on 127.0.0.1:5057..." << std::endl;
    if (!server->start())
    {
        std::cerr << "Failed to start command server" << std::endl;
        juce::shutdownJuce_GUI();
        return 1;
    }

    std::cout << "VST3 Host started. Listening on 127.0.0.1:5057" << std::endl;
    std::cout << "Commands: load_plugin, load_midi, play, stop, get_status, set_parameter, show_editor, hide_editor" << std::endl;
    std::cout.flush();
    std::cerr.flush();

    // Run JUCE dispatch loop on main thread (required for callFunctionOnMessageThread).
    juce::MessageManager::getInstance()->runDispatchLoop();

    // Cleanup
    server.reset();
    pluginMgr.reset();
    juce::shutdownJuce_GUI();
    if (shouldUninitCom)
      CoUninitialize();

    return 0;
}

