# Codebook: Workshop Image-Generation Prompts

This codebook is used to analyse text prompts written by participants during a generative AI workshop on loneliness. Participants were asked to describe photorealistic, everyday scenes of a young adult who might be lonely, written in the third person using observable and contextual cues.

The codebook covers four areas: (A) visual and environmental cues, (B) social and relational cues, (C) AI stereotypes and tropes, and (D) prompt characteristics. A fifth section (E) is reserved for researcher-added codes.

---

## A. Visual and Environmental Cues
*What observable features of the scene signal or suggest loneliness?*

### Code 1: Physical isolation
The prompt describes a scene in which no other people are present or visible — the subject is alone in a space. Apply this code based on scene composition only: does the described scene contain only one person? Code 1 for an empty room, a solitary figure on a street, a person alone at a table. Code 0 if others are present even if the subject is not interacting with them. Do not confuse with Code 12, which concerns explicitly stated social absence rather than scene composition.

### Code 2: Crowded but disconnected
The prompt describes a scene structure in which others are present but the subject is not interacting with them — disconnection is conveyed through spatial or social arrangement. Code 1 when others are nearby but the subject is on the periphery, ignored, or simply not engaging. This code captures scene structure only; it does not require the subject to watch others (Code 13) or to have attempted contact (Code 26). These three codes are distinct and can co-occur: Code 2 = others present and no interaction; Code 13 = subject actively watches others connect; Code 26 = subject tried to connect and failed.

### Code 3: Darkness or dim lighting
The prompt describes low light, darkness, shadows, night-time, or dim/grey conditions as part of the scene's atmosphere.

### Code 4: Empty or sparse environment
The prompt emphasises emptiness, bareness, or absence in the physical space (e.g. empty table, bare walls, vacant seats, deserted public space).

### Code 5: Body language or posture cues
The prompt describes observable body language that signals withdrawal, sadness, or disconnection (e.g. hunched shoulders, arms crossed, slumped posture).

### Code 6: Gaze direction
The prompt specifies where the subject is looking — particularly looking down, staring blankly, looking out a window, or avoiding eye contact with others.

### Code 7: Facial expression specified
The prompt describes a particular facial expression or visible emotional display — for example, crying, a blank or empty stare, tearful eyes, a forced smile, or a neutral/emotionless face. Do not apply this code for general emotional labels ("lonely," "sad," "empty") — those are captured by Code 17. Code this only when a specific observable expression is named.

### Code 8: Technology or screen use
The prompt includes a phone, laptop, or screen as a focal element — either as a substitute for social interaction or as a barrier to it.

### Code 9: Nature or outdoor setting
The scene is set outdoors, in a park, on a bench, by water, or in a natural environment.

### Code 10: Indoor domestic setting
The scene is set inside a home — bedroom, kitchen, living room, or similar private domestic space.

### Code 11: Public or institutional setting
The scene is set in a public or semi-public space such as a café, library, canteen, transport, or classroom.

---

## B. Social and Relational Cues
*What does the prompt imply about the subject's social situation?*

### Code 12: Explicit absence of others
The prompt explicitly states that the subject has no one to be with, has been left out, excluded, or that others are absent in a socially meaningful sense — the social void is named or stated, not just depicted through scene composition. Code 1 for phrases like "no one to sit with," "everyone ignored them," "left out of the group," "nobody noticed." Code 0 if the subject is simply physically alone without the prompt commenting on the social meaning — that is captured by Code 1 instead.

### Code 13: Observing others connecting
The subject is actively watching or is explicitly aware of other people interacting, laughing, bonding, or connecting — the subject is a passive witness to others' connection, in contrast to their own situation. The key distinction from Code 2 is awareness: Code 2 = others are present and not interacting with the subject; Code 13 = the subject notices and observes that others are connecting. Code 0 if the subject is simply in a social setting without the prompt specifying that they observe others. Code 0 if the subject is attempting to connect rather than watching — that is Code 26.

### Code 14: Positive solitude
The prompt frames aloneness as chosen, peaceful, or reflective rather than painful — suggesting the scene is ambiguous or shows solitude rather than loneliness.

### Code 15: Relational loss or transition
The prompt implies a recent change in social situation — moving to a new place, a relationship ending, leaving a group, or a life transition such as starting university.

---

## C. AI Stereotypes and Default Tropes
*Does the prompt reproduce commonly recognised visual shorthand for loneliness that may reflect AI or cultural defaults?*

### Code 16: Rain or grey weather
The prompt includes rain, overcast skies, fog, or explicitly grey or gloomy weather as part of the atmosphere.

### Code 17: Explicit emotional labelling
The prompt uses explicit emotional words such as "lonely," "sad," "isolated," "depressed," or "empty" to describe the subject's inner state, rather than relying on observable cues alone.

### Code 18: Stereotyped figure (elderly person)
The prompt specifies or strongly implies an elderly person as the lone subject — a commonly reproduced visual trope for loneliness.

### Code 19: Stereotyped setting (park bench)
The scene is set on a park bench or equivalent isolated outdoor seat — a widely recognised visual shorthand for loneliness.

---

## D. Prompt Characteristics
*What does the prompt's structure and approach reveal about how participants engaged with the task?*

### Code 20: Highly specific and detailed
The prompt includes concrete, named observable details across at least three of the following six dimensions: (1) setting — a specific named location or room type; (2) lighting — a described light quality or time of day; (3) body language — a named posture or gesture; (4) objects — a specific prop or item mentioned; (5) other people — described characters beyond the subject; (6) atmosphere or sound. Each dimension must be concretely specified, not merely implied. A prompt that says "a sad person in a café" names a setting but provides no detail — code 0. A prompt that says "a young woman at a corner table in a busy university canteen, tray untouched, staring at her phone while groups around her laugh" names setting, objects, body language, and others — code 1.

### Code 21: Minimal or vague
The prompt is either (a) short — strictly under 15 words — or (b) relies primarily on abstract or emotional descriptors without concrete visual detail, regardless of length. A prompt that only says "a lonely person walking alone at night" is vague even if over 15 words because it provides no concrete compositional detail beyond an emotional label and a time of day. Code 0 if the prompt meets the threshold for Code 20. Codes 20 and 21 are mutually exclusive: a prompt cannot be both highly specific and minimal.

### Code 22: Third-person framing maintained
The prompt describes the subject in the third person ("a young woman," "he," "they") as instructed. Code 0 if the prompt is written in first person or is ambiguous.

---

## E. Researcher-Added Codes
*Added following inductive discovery phase (draft_codebook.md). Three themes surfaced by the LLM that were absent from the original deductive codebook.*

### Code 23: Academic setting
The context for loneliness is explicitly a university or academic environment — locations such as classrooms, libraries, lecture halls, student housing, or campus grounds. The character is often identified as a student. Code 1 if the prompt mentions a university, college, or specific academic location; code 0 if the setting is non-academic or unspecified.

### Code 24: Isolation from perceived difference
Loneliness is linked to the character's identity — such as race, culture, gender, or feeling out of sync with peers' life stage — which makes them feel like an outsider in a homogenous group. Code 1 if the prompt explicitly connects the character's isolation to an identity marker that differentiates them from those around them; code 0 if the cause of isolation is situational, environmental, or unspecified.

### Code 25: Failed attempt at connection
The character actively but unsuccessfully tries to engage with others. This includes hesitation before approaching someone, making a social bid that is ignored or rejected, or being unable to sustain a conversation. Code 1 if the prompt describes an active, failed attempt to connect with others; code 0 if the character's isolation is passive or no attempt to engage is described. Distinguish from Code 2 (others present, no interaction — passive) and Code 13 (subject watches others connect — passive awareness). Code 25 requires agency: the subject acts and is rebuffed or ignored.
