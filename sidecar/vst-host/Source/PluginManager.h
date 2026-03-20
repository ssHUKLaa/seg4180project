#pragma once

#include <JuceHeader.h>
#include <memory>
#include <vector>
#include <atomic>

class PluginManager : private juce::AudioIODeviceCallback
{
public:
    PluginManager();
    ~PluginManager();

    juce::StringArray scanForPlugins();
    bool loadPlugin(const juce::File& pluginFile);
    void unloadPlugin();
    bool isPluginLoaded() const { return pluginLoaded.load(); }

    void prepare(double sampleRate, int blockSize);
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages);
    void releaseResources();
    bool loadMidiFile(const juce::File& midiFile);
    bool startPlayback();
    void stopPlayback();
    bool isPlaying() const { return playing.load(); }
    bool isMidiLoaded() const;

    void addMidiEvent(int note, int velocity, int samplePosition, bool isNoteOn);

    juce::Array<juce::AudioProcessorParameter*> getParameters();
    void setParameter(int paramIndex, float value);
    bool hasEditor() const;
    bool showEditor();
    void hideEditor();

    juce::String getPluginName() const;
    juce::String getLastError() const { return lastError; }

private:
    struct ScheduledMidiEvent
    {
        double timeSec = 0.0;
        juce::MidiMessage message;
    };

    juce::AudioPluginFormatManager formatManager;
    juce::KnownPluginList knownPluginList;
    juce::AudioDeviceManager deviceManager;
    std::unique_ptr<juce::AudioProcessor> plugin;
    std::unique_ptr<juce::DocumentWindow> editorWindow;
    std::unique_ptr<juce::AudioProcessorValueTreeState> apvts;
    juce::MidiKeyboardState midiKeyboardState;
    std::vector<ScheduledMidiEvent> scheduledEvents;
    mutable juce::CriticalSection pluginLock;
    juce::CriticalSection transportLock;
    std::atomic<bool> pluginLoaded { false };
    std::atomic<bool> playing { false };
    bool audioDeviceInitialized = false;
    bool audioCallbackAttached = false;
    size_t nextEventIndex = 0;
    double currentSamplePos = 0.0;
    double midiLengthSec = 0.0;
    juce::String lastError;
    juce::File loadedMidiPath;
    double sampleRate = 44100.0;
    int blockSize = 512;

    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                          int numInputChannels,
                                          float* const* outputChannelData,
                                          int numOutputChannels,
                                          int numSamples,
                                          const juce::AudioIODeviceCallbackContext& context) override;
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;

    bool ensureAudioDeviceInitialized();
    void logError(const juce::String& error);
};
