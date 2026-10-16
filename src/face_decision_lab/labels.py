"""Attack taxonomy and the decision questions asked about every face capture."""

BONA_FIDE = "bona_fide"

# Options of the attack_type question. The descriptions are what the model reads, so they say
# what is visible in the capture, not which dataset the sample came from.
ATTACK_TYPES = {
    BONA_FIDE: "A live, unaltered human face captured directly by a camera.",
    "print": "A printed photo of a face held in front of the camera.",
    "replay": "A face shown on a phone, tablet or monitor screen held in front of the camera.",
    "mask": "A physical mask, paper cutout or prosthetic imitating a person's face.",
    "face_swap": "A face digitally replaced with another person's face.",
    "face_reenactment": "A real face digitally driven to change its expression or mouth movement.",
    "fully_synthetic": "A face generated entirely by a generative model, with no real source photo.",
}

QUESTIONS = {
    "is_bona_fide": {
        "type": "noul",
        "instructions": "Is this a live, unaltered human face captured directly by a camera?",
    },
    "attack_type": {
        "type": "choice",
        "instructions": "Which kind of capture is this face?",
        "criteria": ATTACK_TYPES,
    },
}

# Fixed for every row so the text never hints at the label or the source dataset.
STATE = {"input": "One cropped face image from an identity verification capture."}
