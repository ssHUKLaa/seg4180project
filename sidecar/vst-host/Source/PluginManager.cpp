#include "PluginManager.h"
#include <algorithm>
#include <iostream>

namespace
{
class PluginEditorWindow : public juce::DocumentWindow
{
public:
    explicit PluginEditorWindow(const juce::String& title)
        : juce::DocumentWindow(title,
                               juce::Colours::darkgrey,
                               juce::DocumentWindow::closeButton,
                               true)
    {
    }

    void closeButtonPressed() override
    {
        setVisible(false);
    }
};
}

PluginManager::PluginManager()
{
    formatManager.addDefaultFormats();
}

PluginManager::~PluginManager()
{
    if (audioCallbackAttached)
    {
        deviceManager.removeAudioCallback(this);
        audioCallbackAttached = false;
    }
    stopPlayback();
    releaseResources();
}

juce::StringArray PluginManager::scanForPlugins()
{
    juce::StringArray plugins;
    
    // Get standard plugin directories
    juce::Array<juce::File> pluginSearchPaths;
    
    // Windows VST3 directory
    juce::File vst3Dir (juce::File::getSpecialLocation(
        juce::File::commonApplicationDataDirectory)
        .getChildFile("VST3"));
    
    if (vst3Dir.exists())
        pluginSearchPaths.add(vst3Dir);
    
    // VST2 directory (legacy)
    juce::File vst2Dir (juce::File::getSpecialLocation(
        juce::File::commonApplicationDataDirectory)
        .getChildFile("Steinberg/VSTPlugins"));
    
    if (vst2Dir.exists())
        pluginSearchPaths.add(vst2Dir);
    
    // Scan for .vst3 files
    for (const auto& dir : pluginSearchPaths)
    {
        juce::Array<juce::File> foundFiles;
        dir.findChildFiles(foundFiles, juce::File::findFiles, true, "*.vst3");
        
        for (const auto& file : foundFiles)
            plugins.add(file.getFullPathName());
    }
    
    return plugins;
}

bool PluginManager::loadPlugin(const juce::File& pluginFile)
{
    std::cout << "PluginManager: loadPlugin begin: " << pluginFile.getFullPathName() << std::endl;
    std::cout.flush();
    juce::Logger::writeToLog("PluginManager: loadPlugin begin: " + pluginFile.getFullPathName());
    unloadPlugin();

    if (!pluginFile.exists())
    {
        logError("Plugin file not found: " + pluginFile.getFullPathName());
        return false;
    }

    juce::String error;
    std::unique_ptr<juce::AudioPluginInstance> instance;

    juce::OwnedArray<juce::PluginDescription> foundTypes;
    std::cout << "PluginManager: loadPlugin scanning formats" << std::endl;
    std::cout.flush();
    juce::Logger::writeToLog("PluginManager: loadPlugin scanning formats");

    for (auto* format : formatManager.getFormats())
    {
        if (format == nullptr)
            continue;

        if (!format->fileMightContainThisPluginType(pluginFile.getFullPathName()))
            continue;

        std::cout << "PluginManager: probing format: " << format->getName() << std::endl;
        std::cout.flush();
        juce::Logger::writeToLog("PluginManager: probing format: " + format->getName());
        format->findAllTypesForFile(foundTypes, pluginFile.getFullPathName());
        if (foundTypes.size() > 0)
            break;
    }

    if (foundTypes.isEmpty())
    {
        logError("No plugin types found in: " + pluginFile.getFullPathName());
        return false;
    }

    auto* desc = foundTypes.getFirst();
    if (desc == nullptr)
    {
        logError("Plugin description missing for: " + pluginFile.getFullPathName());
        return false;
    }

    std::cout << "PluginManager: creating plugin instance via scanned description: " << desc->name << std::endl;
    std::cout.flush();
    juce::Logger::writeToLog("PluginManager: creating plugin instance: " + desc->name);

    instance = formatManager.createPluginInstance(*desc, sampleRate, blockSize, error);

    if (instance == nullptr)
    {
        if (error.isEmpty())
            error = "Unknown JUCE error while creating plugin instance";
        logError("Failed to create plugin instance: " + error + " (" + pluginFile.getFullPathName() + ")");
        return false;
    }

    {
        const juce::ScopedLock lock(pluginLock);
        plugin = std::move(instance);
        std::cout << "PluginManager: configuring buses/rate/buffer" << std::endl;
        std::cout.flush();
        juce::Logger::writeToLog("PluginManager: configuring buses/rate/buffer");

        juce::AudioProcessor::BusesLayout layout;
        if (plugin->getTotalNumOutputChannels() >= 2)
            layout.outputBuses.add(juce::AudioChannelSet::stereo());
        else if (plugin->getTotalNumOutputChannels() == 1)
            layout.outputBuses.add(juce::AudioChannelSet::mono());

        if (layout.outputBuses.size() > 0 && plugin->checkBusesLayoutSupported(layout))
            plugin->setBusesLayout(layout);

        plugin->setRateAndBufferSizeDetails(sampleRate, blockSize);
        plugin->prepareToPlay(sampleRate, blockSize);
    }

    pluginLoaded.store(true);

    lastError.clear();
    std::cout << "PluginManager: loaded plugin " << plugin->getName() << std::endl;
    std::cout.flush();
    juce::Logger::writeToLog("PluginManager: loaded plugin " + plugin->getName());
    return true;
}

void PluginManager::unloadPlugin()
{
    pluginLoaded.store(false);
    stopPlayback();

    if (audioCallbackAttached)
    {
        deviceManager.removeAudioCallback(this);
        audioCallbackAttached = false;
    }

    hideEditor();

    {
        const juce::ScopedLock lock(pluginLock);
        if (plugin)
            plugin->releaseResources();

        plugin.reset();
        apvts.reset();
    }
}

void PluginManager::prepare(double sr, int bs)
{
    sampleRate = sr;
    blockSize = bs;
    
    const juce::ScopedLock lock(pluginLock);
    if (plugin)
    {
        juce::AudioProcessor::BusesLayout layout;
        layout.outputBuses.add(juce::AudioChannelSet::stereo());
        
        if (plugin->getBusesLayout() != layout)
            plugin->setBusesLayout(layout);

        plugin->setRateAndBufferSizeDetails(sampleRate, blockSize);
        plugin->prepareToPlay(sampleRate, blockSize);
    }
}

void PluginManager::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    const juce::ScopedLock lock(pluginLock);
    if (!plugin)
        return;
    
    plugin->processBlock(buffer, midiMessages);
}

void PluginManager::releaseResources()
{
    const juce::ScopedLock lock(pluginLock);
    if (plugin)
        plugin->releaseResources();
}

bool PluginManager::loadMidiFile(const juce::File& midiFile)
{
    if (!midiFile.existsAsFile())
    {
        logError("MIDI file not found: " + midiFile.getFullPathName());
        return false;
    }

    juce::FileInputStream stream(midiFile);
    if (!stream.openedOk())
    {
        logError("Failed to open MIDI file: " + midiFile.getFullPathName());
        return false;
    }

    juce::MidiFile mf;
    if (!mf.readFrom(stream))
    {
        logError("Failed to parse MIDI file: " + midiFile.getFullPathName());
        return false;
    }

    mf.convertTimestampTicksToSeconds();

    std::vector<ScheduledMidiEvent> newEvents;
    newEvents.reserve(2048);

    double maxTime = 0.0;
    for (int t = 0; t < mf.getNumTracks(); ++t)
    {
        auto* seq = mf.getTrack(t);
        if (seq == nullptr)
            continue;

        for (int i = 0; i < seq->getNumEvents(); ++i)
        {
            auto* ev = seq->getEventPointer(i);
            if (ev == nullptr)
                continue;

            const auto msg = ev->message;
            if (msg.isMetaEvent())
                continue;

            const double ts = msg.getTimeStamp();
            if (ts < 0.0)
                continue;

            ScheduledMidiEvent item;
            item.timeSec = ts;
            item.message = msg;
            newEvents.push_back(std::move(item));
            maxTime = juce::jmax(maxTime, ts);
        }
    }

    std::sort(newEvents.begin(), newEvents.end(), [](const ScheduledMidiEvent& a, const ScheduledMidiEvent& b)
    {
        return a.timeSec < b.timeSec;
    });

    {
        const juce::ScopedLock lock(transportLock);
        scheduledEvents = std::move(newEvents);
        nextEventIndex = 0;
        currentSamplePos = 0.0;
        midiLengthSec = maxTime;
        loadedMidiPath = midiFile;
        playing.store(false);
    }

    return true;
}

bool PluginManager::startPlayback()
{
    if (!isPluginLoaded())
    {
        logError("Cannot play: no plugin loaded");
        return false;
    }

    const juce::ScopedLock lock(transportLock);
    if (scheduledEvents.empty())
    {
        logError("Cannot play: no MIDI loaded");
        return false;
    }

    nextEventIndex = 0;
    currentSamplePos = 0.0;
    playing.store(true);

    if (!ensureAudioDeviceInitialized())
    {
        playing.store(false);
        return false;
    }

    if (!audioCallbackAttached)
    {
        deviceManager.addAudioCallback(this);
        audioCallbackAttached = true;
    }

    return true;
}

void PluginManager::stopPlayback()
{
    playing.store(false);
}

bool PluginManager::isMidiLoaded() const
{
    return !scheduledEvents.empty();
}

void PluginManager::addMidiEvent(int note, int velocity, int samplePosition, bool isNoteOn)
{
    juce::ignoreUnused(samplePosition);

    // This will be connected to the MIDI router
    // For now, just maintain state
    if (isNoteOn)
        midiKeyboardState.noteOn(1, note, velocity / 127.0f);
    else
        midiKeyboardState.noteOff(1, note, velocity / 127.0f);
}

juce::Array<juce::AudioProcessorParameter*> PluginManager::getParameters()
{
    juce::Array<juce::AudioProcessorParameter*> params;
    const juce::ScopedLock lock(pluginLock);
    if (plugin)
    {
        for (auto* param : plugin->getParameters())
            params.add(param);
    }
    return params;
}

void PluginManager::setParameter(int paramIndex, float value)
{
    const juce::ScopedLock lock(pluginLock);
    if (plugin && paramIndex >= 0 && paramIndex < plugin->getParameters().size())
        plugin->getParameters()[paramIndex]->setValue(value);
}

bool PluginManager::hasEditor() const
{
    const juce::ScopedLock lock(pluginLock);
    return plugin != nullptr && plugin->hasEditor();
}

bool PluginManager::showEditor()
{
    juce::AudioProcessor* proc = nullptr;
    {
        const juce::ScopedLock lock(pluginLock);
        proc = plugin.get();
    }

    if (proc == nullptr)
    {
        logError("No plugin loaded");
        return false;
    }

    if (!proc->hasEditor())
    {
        logError("Loaded plugin has no editor");
        return false;
    }

    if (editorWindow == nullptr)
    {
        juce::AudioProcessorEditor* editor = nullptr;
        constexpr int maxAttempts = 6;
        for (int attempt = 1; attempt <= maxAttempts && editor == nullptr; ++attempt)
        {
            editor = proc->createEditorIfNeeded();
            if (editor != nullptr)
                break;

            std::cout << "PluginManager: editor creation attempt " << attempt << " failed" << std::endl;
            std::cout.flush();
            juce::Logger::writeToLog("PluginManager: editor creation attempt failed: " + juce::String(attempt));
            juce::Thread::sleep(120);
        }

        if (editor == nullptr)
        {
            logError("Failed to create plugin editor after retries");
            return false;
        }

        auto window = std::make_unique<PluginEditorWindow>(proc->getName());
        window->setUsingNativeTitleBar(true);
        window->setResizable(true, true);
        window->setContentOwned(editor, true);
        window->centreWithSize(editor->getWidth(), editor->getHeight());
        editorWindow = std::move(window);
    }

    editorWindow->setVisible(true);
    editorWindow->toFront(true);
    return true;
}

void PluginManager::hideEditor()
{
    const juce::ScopedLock lock(pluginLock);
    if (editorWindow != nullptr)
    {
        editorWindow->setVisible(false);
        editorWindow.reset();
    }
}

juce::String PluginManager::getPluginName() const
{
    const juce::ScopedLock lock(pluginLock);
    return plugin ? plugin->getName() : "No plugin loaded";
}

void PluginManager::logError(const juce::String& error)
{
    lastError = error;
    juce::Logger::writeToLog("PluginManager: " + error);
}

void PluginManager::audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                                     int numInputChannels,
                                                     float* const* outputChannelData,
                                                     int numOutputChannels,
                                                     int numSamples,
                                                     const juce::AudioIODeviceCallbackContext& context)
{
    juce::ignoreUnused(inputChannelData, numInputChannels, context);

    for (int ch = 0; ch < numOutputChannels; ++ch)
    {
        if (outputChannelData[ch] != nullptr)
            juce::FloatVectorOperations::clear(outputChannelData[ch], numSamples);
    }

    if (numOutputChannels <= 0)
        return;

    juce::AudioBuffer<float> outBuffer(outputChannelData, numOutputChannels, numSamples);
    juce::MidiBuffer midi;

    {
        const juce::ScopedLock lock(transportLock);

        if (playing.load() && !scheduledEvents.empty() && sampleRate > 0.0)
        {
            const double blockStartSec = currentSamplePos / sampleRate;
            const double blockEndSec = (currentSamplePos + numSamples) / sampleRate;

            while (nextEventIndex < scheduledEvents.size())
            {
                const auto& ev = scheduledEvents[nextEventIndex];
                if (ev.timeSec >= blockEndSec)
                    break;

                const int sampleOffset = juce::jlimit(
                    0,
                    juce::jmax(0, numSamples - 1),
                    (int) ((ev.timeSec - blockStartSec) * sampleRate)
                );

                midi.addEvent(ev.message, sampleOffset);
                ++nextEventIndex;
            }

            currentSamplePos += numSamples;
            if (nextEventIndex >= scheduledEvents.size())
                playing.store(false);
        }
    }

    {
        const juce::ScopedLock lock(pluginLock);
        if (plugin == nullptr)
            return;
        plugin->processBlock(outBuffer, midi);
    }
}

void PluginManager::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    if (device == nullptr)
        return;

    sampleRate = device->getCurrentSampleRate();
    blockSize = device->getCurrentBufferSizeSamples();

    const juce::ScopedLock lock(pluginLock);
    if (plugin != nullptr)
    {
        plugin->setRateAndBufferSizeDetails(sampleRate, blockSize);
        plugin->prepareToPlay(sampleRate, blockSize);
    }
}

void PluginManager::audioDeviceStopped()
{
    stopPlayback();
}

bool PluginManager::ensureAudioDeviceInitialized()
{
    if (audioDeviceInitialized)
        return true;

    auto err = deviceManager.initialiseWithDefaultDevices(0, 2);
    if (err.isNotEmpty())
    {
        logError("Audio device init failed: " + err);
        return false;
    }

    audioDeviceInitialized = true;
    return true;
}
