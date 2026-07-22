# Kazakh Calm Teacher Voice Design

## Goal

Make all production Kazakh OmniVoice speech use a calm teacher delivery while
keeping the existing young male voice.

## Design

Change the default OmniVoice instruction to:

`male, young adult, calm teacher, clear articulation, moderate pitch`

Keep the profile fixed at service startup. Do not add per-request intonation
controls, new API fields, or dependencies. Update both the Python default and
the Docker Compose default so local and production behavior match.

## Deployment

Production does not override `OMNIVOICE_INSTRUCT` in `.env`, so rebuilding only
the `voice-omnivoice` service will activate the new profile globally. Verify the
health endpoint reports the new profile, synthesize the Kazakh introduction,
and replace the local Kazakh MP3 sample with that production output.

## Testing

Add one unit assertion for the default profile, run the focused test, run the
full test suite, validate the rendered Compose configuration, and verify a
production synthesis returns non-empty WAV audio.
