"""Prompts for the avatar states, in the structured caption style EchoMimicV3 was trained on (see third_party/echomimic_v3/
datasets/echomimicv3_demos/prompts). Each state also says where its audio comes from: the quiet states are driven by silence so
the mouth stays closed, the speaking state by real speech.

`seed` is the state's own base seed (the CLI --seed is added to it), so two avatars generated with the same settings differ
only by their image.
"""

_BACKGROUND = ("Background: Preserve background integrity matching the reference image's spatial configuration, lighting conditions, "
               "and color temperature.")

STATES = {
    "idle": {
        "audio": "silence",
        "seed": 11,
        "template": (
            "Body Positions and Posture: {subject} stands relaxed and upright, facing the camera. The chest and shoulders rise and "
            "fall very slightly with calm breathing and the weight shifts a little, so the whole body sways gently. "
            "Hands and Fingers Movements: The arms hang naturally and stay relaxed with only tiny movements; the fingers are loose. "
            "Face Expressions Change: A calm, friendly, neutral expression with the mouth closed; the eyes blink slowly now and then. "
            "Eyes Movement: The eyes look steadily at the camera. Head Movements: The head stays nearly still with a very subtle "
            "natural sway. {background} "
            "Overall Description: The person waits calmly and attentively with the mouth closed, moving only with natural breathing."
        ),
    },
    "listening": {
        "audio": "silence",
        "seed": 12,
        "template": (
            "Body Positions and Posture: {subject} listens attentively to someone who is speaking, facing the camera, standing "
            "upright with relaxed shoulders that move subtly with breathing. "
            "Hands and Fingers Movements: The arms stay relaxed and the hands barely move. "
            "Face Expressions Change: An attentive, interested expression with the mouth closed; the eyes blink naturally. "
            "Eyes Movement: The eyes stay on the camera. Head Movements: The head gives a slow, small nod and then tilts very "
            "slightly to one side, as if following what is being said, and returns to the centre. {background} "
            "Overall Description: The person is listening carefully, mouth closed, with small natural nods."
        ),
    },
    "thinking": {
        "audio": "silence",
        "seed": 13,
        "template": (
            "Body Positions and Posture: {subject} pauses to think about a question, standing upright and facing the camera with "
            "relaxed shoulders. "
            "Hands and Fingers Movements: The arms stay relaxed and the hands barely move. "
            "Face Expressions Change: A thoughtful expression with the mouth closed, the eyebrows rising slightly. "
            "Eyes Movement: The eyes glance briefly up and to the side for a moment, then return to the camera. "
            "Head Movements: The head tilts slightly to one side while thinking and then comes back to the centre. {background} "
            "Overall Description: The person is thinking for a moment before answering, mouth closed, with small natural movements."
        ),
    },
    "speaking": {
        "audio": "speech",
        "seed": 14,
        "template": (
            "Body Positions and Posture: {subject} talks to the camera in a warm, engaging way, standing upright; the shoulders and "
            "torso move naturally with the rhythm of the speech. "
            "Hands and Fingers Movements: Both hands rise to chest height and make open, expressive gestures in time with the "
            "speech, with open palms and small smooth movements, then relax back down. The fingers move naturally. "
            "Face Expressions Change: A friendly, expressive face with the mouth moving as in natural speech and an occasional "
            "smile. Eyes Movement: The eyes look at the camera. Head Movements: The head nods gently for emphasis and tilts "
            "slightly with the speech. {background} "
            "Overall Description: The person speaks to the camera with natural, relaxed hand gestures and body movement."
        ),
    },
}


def build_prompt(state, subject=None):
    subject = (subject or "The person").strip()
    return STATES[state]["template"].format(subject=subject[0].upper() + subject[1:], background=_BACKGROUND)
