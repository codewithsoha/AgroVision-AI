"""Editable treatment and prevention reference for the PlantVillage classes.

Keys are the PlantVillage folder names, including the dataset's original
spelling. To edit guidance, change the text in DISEASES; no other code changes.

This is general educational information. Fungicide and pesticide registration,
dose, and safety rules vary by country and crop, so chemical entries name active
ingredients only, without doses, and always defer to the product label.
"""

from __future__ import annotations

from dataclasses import dataclass

DISCLAIMER = (
    "General educational guidance, not professional agricultural advice. No "
    "treatment is guaranteed to work. Confirm the diagnosis and check locally "
    "approved products with an agricultural extension officer before treating."
)


@dataclass(frozen=True)
class DiseaseInfo:
    name: str
    cause: str
    summary: str
    organic: tuple[str, ...]
    chemical: tuple[str, ...]
    prevention: tuple[str, ...]


HEALTHY_GUIDANCE = DiseaseInfo(
    name="Healthy",
    cause="No disease class",
    summary="This class represents leaves without a visible disease in the dataset.",
    organic=("No treatment is needed for a healthy plant.",),
    chemical=("Avoid unnecessary sprays; they add cost and can harm beneficial organisms.",),
    prevention=(
        "Scout plants regularly so that problems are noticed early.",
        "Water at the base of the plant and keep foliage as dry as practical.",
        "Maintain good spacing, balanced nutrition, and clean tools.",
    ),
)

HEALTHY_CLASSES = (
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___healthy",
    "Grape___healthy",
    "Peach___healthy",
    "Pepper,_bell___healthy",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Strawberry___healthy",
    "Tomato___healthy",
)

_LATE_BLIGHT = dict(
    name="Late blight",
    cause="Oomycete: Phytophthora infestans",
    summary=(
        "A fast-spreading disease of potato and tomato that causes dark, "
        "water-soaked lesions and thrives in cool, wet weather."
    ),
    chemical=(
        "Protectants such as mancozeb or chlorothalonil, applied before infection where approved.",
        "Systemic options such as cymoxanil or metalaxyl; rotate modes of action, as resistance is known.",
    ),
)

DISEASES: dict[str, DiseaseInfo] = {
    "Apple___Apple_scab": DiseaseInfo(
        name="Apple scab",
        cause="Fungus: Venturia inaequalis",
        summary="Causes olive-brown, velvety spots on leaves and scabby lesions on fruit.",
        organic=(
            "Rake and compost or shred fallen leaves, where the fungus overwinters.",
            "Sulfur or copper sprays, as labeled for the crop.",
        ),
        chemical=("Protectants such as captan or mancozeb.", "Group 3 fungicides such as myclobutanil; rotate to limit resistance."),
        prevention=("Plant scab-resistant cultivars.", "Prune to improve airflow and leaf drying."),
    ),
    "Apple___Black_rot": DiseaseInfo(
        name="Black rot",
        cause="Fungus: Diplodia seriata (Botryosphaeria obtusa)",
        summary="Causes 'frogeye' leaf spots, fruit rot, and cankers on limbs.",
        organic=(
            "Prune out dead wood and cankers during dry weather.",
            "Remove mummified fruit from trees and the ground.",
        ),
        chemical=("Fungicides such as captan or thiophanate-methyl, as labeled.",),
        prevention=("Keep trees vigorous and avoid wounding.", "Remove dead trees and wood piles near the orchard."),
    ),
    "Apple___Cedar_apple_rust": DiseaseInfo(
        name="Cedar apple rust",
        cause="Fungus: Gymnosporangium juniperi-virginianae",
        summary="Causes bright yellow-orange leaf spots; the fungus alternates between apple and juniper hosts.",
        organic=(
            "Remove galls from nearby junipers before spring spore release.",
            "Sulfur sprays may give limited protection when applied early.",
        ),
        chemical=("Fungicides such as myclobutanil or mancozeb, timed from pink bud onward.",),
        prevention=("Plant rust-resistant apple cultivars.", "Avoid planting junipers close to apple trees."),
    ),
    "Cherry_(including_sour)___Powdery_mildew": DiseaseInfo(
        name="Powdery mildew",
        cause="Fungus: Podosphaera clandestina",
        summary="Forms white, powdery growth on young leaves and shoots.",
        organic=("Sulfur or potassium bicarbonate sprays, as labeled.", "Neem oil may help at an early stage."),
        chemical=("Fungicides such as myclobutanil or trifloxystrobin; rotate modes of action.",),
        prevention=("Prune to open the canopy.", "Avoid excess nitrogen, which favors soft new growth."),
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": DiseaseInfo(
        name="Cercospora / gray leaf spot",
        cause="Fungus: Cercospora zeae-maydis",
        summary="Causes long, rectangular gray-tan lesions running between leaf veins.",
        organic=("Rotate away from corn for at least a year.", "Bury or manage infected residue where practical."),
        chemical=("Strobilurin and triazole fungicides around tasseling, when disease pressure justifies the cost.",),
        prevention=("Choose hybrids with gray leaf spot tolerance.", "Avoid continuous corn in high-residue fields."),
    ),
    "Corn_(maize)___Common_rust_": DiseaseInfo(
        name="Common rust",
        cause="Fungus: Puccinia sorghi",
        summary="Produces small, reddish-brown pustules on both leaf surfaces.",
        organic=("Usually managed through resistant hybrids rather than sprays.", "Monitor; mild infections often need no action."),
        chemical=("Triazole or strobilurin fungicides if severe before tasseling.",),
        prevention=("Plant resistant hybrids.", "Plant early where this avoids peak rust periods."),
    ),
    "Corn_(maize)___Northern_Leaf_Blight": DiseaseInfo(
        name="Northern leaf blight",
        cause="Fungus: Exserohilum turcicum",
        summary="Causes long, cigar-shaped gray-green to tan lesions.",
        organic=("Rotate crops and manage infected residue.",),
        chemical=("Triazole or strobilurin fungicides around tasseling, when disease pressure justifies the cost.",),
        prevention=("Plant resistant hybrids.", "Avoid continuous corn in previously infected fields."),
    ),
    "Grape___Black_rot": DiseaseInfo(
        name="Black rot",
        cause="Fungus: Guignardia bidwellii",
        summary="Causes tan leaf spots with dark borders and shrivels berries into hard, black mummies.",
        organic=("Remove mummified berries and infected canes.", "Copper or sulfur sprays offer limited protection."),
        chemical=("Fungicides such as captan, mancozeb, or myclobutanil, focused from bloom through early fruit growth.",),
        prevention=("Train and prune vines for airflow.", "Control weeds to keep the canopy dry."),
    ),
    "Grape___Esca_(Black_Measles)": DiseaseInfo(
        name="Esca (black measles)",
        cause="Complex of wood-infecting fungi, including Phaeomoniella chlamydospora",
        summary="A long-term trunk disease that causes striped leaves and spotted berries, with no reliable cure.",
        organic=(
            "Mark affected vines and consider removing badly infected wood.",
            "Biological wound protectants based on Trichoderma may reduce new infections.",
        ),
        chemical=("No reliably curative product is known; the focus is on protecting pruning wounds.",),
        prevention=("Prune in dry weather and protect large cuts.", "Start with healthy planting material."),
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": DiseaseInfo(
        name="Leaf blight (Isariopsis leaf spot)",
        cause="Fungus: Pseudocercospora vitis",
        summary="Causes irregular brown spots, usually late in the season.",
        organic=("Remove fallen infected leaves.", "Improve canopy airflow through pruning."),
        chemical=("Copper or mancozeb protectants, as labeled; programs for other grape diseases often also help.",),
        prevention=("Keep vines well-trained and balanced.", "Avoid prolonged leaf wetness."),
    ),
    "Orange___Haunglongbing_(Citrus_greening)": DiseaseInfo(
        name="Huanglongbing (citrus greening)",
        cause="Bacteria: Candidatus Liberibacter species, spread by citrus psyllids",
        summary="A severe, incurable citrus disease causing blotchy yellow leaves and bitter, misshapen fruit.",
        organic=(
            "There is no cure; confirm by laboratory test and follow local guidance on removing infected trees.",
            "Encourage natural enemies of the psyllid.",
        ),
        chemical=("Chemicals do not cure trees; insecticides are used only to manage psyllids under local programs.",),
        prevention=("Use certified disease-free nursery trees.", "Report suspected cases; the disease is regulated in many regions."),
    ),
    "Peach___Bacterial_spot": DiseaseInfo(
        name="Bacterial spot",
        cause="Bacterium: Xanthomonas arboricola pv. pruni",
        summary="Causes angular leaf spots, 'shot holes', and pitted fruit.",
        organic=("Copper around dormancy, as labeled; in-season copper can damage peach leaves.",),
        chemical=("Oxytetracycline where registered for this use.",),
        prevention=("Plant tolerant cultivars.", "Avoid excess nitrogen and sites exposed to wind-driven rain."),
    ),
    "Pepper,_bell___Bacterial_spot": DiseaseInfo(
        name="Bacterial spot",
        cause="Bacteria: Xanthomonas species",
        summary="Causes small, water-soaked spots that turn brown and can make leaves drop.",
        organic=("Remove and destroy badly infected plants.", "Copper sprays, as labeled, may slow spread."),
        chemical=("Copper combined with mancozeb, where approved.",),
        prevention=("Use disease-free seed and transplants.", "Rotate crops and avoid working among wet plants."),
    ),
    "Potato___Early_blight": DiseaseInfo(
        name="Early blight",
        cause="Fungus: Alternaria solani",
        summary="Causes dark spots with concentric rings, first on older leaves.",
        organic=("Remove infected lower leaves.", "Copper sprays, as labeled, may slow spread."),
        chemical=("Fungicides such as mancozeb, chlorothalonil, or azoxystrobin; rotate modes of action.",),
        prevention=("Rotate away from potato and tomato.", "Keep plants well nourished to reduce stress."),
    ),
    "Potato___Late_blight": DiseaseInfo(
        **_LATE_BLIGHT,
        organic=("Remove and destroy infected plants promptly; do not compost them.", "Copper sprays, as labeled, can offer some protection."),
        prevention=("Plant certified seed potatoes.", "Hill soil over tubers and destroy cull piles."),
    ),
    "Squash___Powdery_mildew": DiseaseInfo(
        name="Powdery mildew",
        cause="Fungus: Podosphaera xanthii",
        summary="Forms white, powdery patches on leaves and stems.",
        organic=("Potassium bicarbonate or neem oil, as labeled.", "Sulfur can help, but some squash varieties are sensitive; test first."),
        chemical=("Fungicides such as chlorothalonil or myclobutanil; rotate modes of action.",),
        prevention=("Plant resistant varieties.", "Space plants for airflow and sunlight."),
    ),
    "Strawberry___Leaf_scorch": DiseaseInfo(
        name="Leaf scorch",
        cause="Fungus: Diplocarpon earlianum",
        summary="Causes small purple spots that merge until leaves look scorched.",
        organic=("Remove old infected leaves after harvest.", "Keep beds free of excess foliage and weeds."),
        chemical=("Fungicides such as captan or myclobutanil, as labeled.",),
        prevention=("Use healthy planting stock and resistant varieties.", "Prefer drip irrigation and renew old beds."),
    ),
    "Tomato___Bacterial_spot": DiseaseInfo(
        name="Bacterial spot",
        cause="Bacteria: Xanthomonas species",
        summary="Causes small, dark, greasy-looking spots on leaves and fruit.",
        organic=("Remove infected leaves in dry conditions.", "Copper sprays, as labeled, may slow spread."),
        chemical=("Copper combined with mancozeb, where approved.",),
        prevention=("Use disease-free seed and transplants.", "Rotate crops and avoid overhead watering."),
    ),
    "Tomato___Early_blight": DiseaseInfo(
        name="Early blight",
        cause="Fungi: Alternaria species",
        summary="Causes target-like brown spots, first on older, lower leaves.",
        organic=("Remove infected lower leaves.", "Mulch to reduce soil splash onto foliage."),
        chemical=("Fungicides such as chlorothalonil, mancozeb, or azoxystrobin; rotate modes of action.",),
        prevention=("Rotate away from tomato and potato.", "Stake plants and water at the base."),
    ),
    "Tomato___Late_blight": DiseaseInfo(
        **_LATE_BLIGHT,
        organic=("Remove and destroy infected plants promptly; do not compost them.", "Copper sprays, as labeled, can offer some protection."),
        prevention=("Use resistant varieties where available.", "Keep foliage dry and remove volunteer potato and tomato plants."),
    ),
    "Tomato___Leaf_Mold": DiseaseInfo(
        name="Leaf mold",
        cause="Fungus: Passalora fulva",
        summary="Causes pale spots above with olive-gray mold beneath; common in humid greenhouses.",
        organic=("Lower humidity and increase ventilation.", "Remove infected leaves carefully."),
        chemical=("Protectants such as chlorothalonil or mancozeb, as labeled.",),
        prevention=("Grow resistant varieties.", "Space and prune plants for airflow."),
    ),
    "Tomato___Septoria_leaf_spot": DiseaseInfo(
        name="Septoria leaf spot",
        cause="Fungus: Septoria lycopersici",
        summary="Causes many small spots with gray centers and dark edges.",
        organic=("Remove infected lower leaves.", "Mulch to reduce spores splashing up from soil."),
        chemical=("Fungicides such as chlorothalonil or mancozeb, as labeled.",),
        prevention=("Rotate crops and clear plant debris.", "Avoid overhead watering."),
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": DiseaseInfo(
        name="Two-spotted spider mite",
        cause="Pest, not a disease: Tetranychus urticae",
        summary="Tiny mites feed on leaves, causing yellow stippling and fine webbing in hot, dry conditions.",
        organic=("Dislodge mites with a strong water spray.", "Insecticidal soap or horticultural oil; predatory mites can also help."),
        chemical=("Miticides such as abamectin or bifenazate; rotate modes of action.",),
        prevention=("Avoid plant water stress and dust.", "Avoid broad-spectrum insecticides, which kill natural predators."),
    ),
    "Tomato___Target_Spot": DiseaseInfo(
        name="Target spot",
        cause="Fungus: Corynespora cassiicola",
        summary="Causes brown spots with faint rings on leaves and fruit.",
        organic=("Remove infected lower leaves and debris.", "Improve airflow through pruning and staking."),
        chemical=("Fungicides such as chlorothalonil or azoxystrobin; rotate modes of action.",),
        prevention=("Rotate crops and clear plant debris.", "Avoid long periods of leaf wetness."),
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": DiseaseInfo(
        name="Tomato yellow leaf curl virus",
        cause="Virus: begomovirus spread by the whitefly Bemisia tabaci",
        summary="Causes yellow, upward-curling leaves and stunted plants, with no cure for infected plants.",
        organic=("Remove infected plants to reduce spread.", "Use insect netting and reflective mulch to deter whiteflies."),
        chemical=("No chemical cures the virus; insecticides only target whiteflies and should follow local resistance guidance.",),
        prevention=("Plant resistant varieties.", "Use whitefly-free transplants and control weed hosts."),
    ),
    "Tomato___Tomato_mosaic_virus": DiseaseInfo(
        name="Tomato mosaic virus",
        cause="Virus: tobamovirus spread by contact, tools, and seed",
        summary="Causes mottled, light and dark green leaves and distorted growth, with no cure for infected plants.",
        organic=("Remove and destroy infected plants.", "Disinfect tools and wash hands before handling plants."),
        chemical=("No chemical cures this virus.",),
        prevention=("Use certified seed and resistant varieties.", "Avoid handling healthy plants after infected ones."),
    ),
}

ALL_CLASSES = tuple(sorted((*DISEASES, *HEALTHY_CLASSES)))


def parse_class_name(class_name: str) -> tuple[str, str, bool]:
    """Split a folder name such as 'Tomato___Late_blight' for display.

    Returns (crop, condition, is_healthy).
    """
    crop, _, condition = class_name.partition("___")
    crop = crop.replace("_", " ").strip()
    is_healthy = condition.lower() == "healthy"
    if is_healthy:
        return crop, "Healthy", True
    if class_name in DISEASES:
        return crop, DISEASES[class_name].name, False
    return crop, condition.replace("_", " ").strip() or "Unknown", False


def get_recommendation(class_name: str) -> DiseaseInfo | None:
    """Return guidance for a class, or None if the class has no entry."""
    if class_name in DISEASES:
        return DISEASES[class_name]
    if class_name in HEALTHY_CLASSES:
        return HEALTHY_GUIDANCE
    return None
