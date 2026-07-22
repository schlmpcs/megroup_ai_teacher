# Kazakh Calm Teacher Voice Design

## Goal

Make all production Kazakh OmniVoice speech use a calm teacher delivery while
keeping the existing young male voice.

## Design

OmniVoice accepts only a fixed vocabulary, so use its closest supported calm
profile:

`male, young adult, low pitch`

Keep the profile fixed at service startup. Do not add per-request intonation
controls, new API fields, or dependencies. Send Kazakh OmniVoice requests at
speed `0.9` while leaving other TTS backends at `1.0`. Update both the Python
default and the Docker Compose default so local and production behavior match.

## Deployment

Production does not override `OMNIVOICE_INSTRUCT` in `.env`, so rebuilding the
`voice-omnivoice` service activates the new profile. Rebuild the API as well so
it sends speed `0.9`. Verify the health endpoint reports the new profile,
synthesize the Kazakh introduction, and replace the local Kazakh MP3 sample with
that production output.

## Testing

Add one unit assertion for the default profile, run the focused test, run the
full test suite, validate the rendered Compose configuration, and verify a
production synthesis returns non-empty WAV audio.
