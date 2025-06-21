import FormPl from "./formpl";
import TransitionBackground from "../components/TransitionBackground";
export default function Home() {
  return (
    <main>
    <TransitionBackground gradient="bg-gradient-to-br from-blue-400 to-white transition-colors duration-700">
      <FormPl></FormPl>
      </TransitionBackground>
    </main>
  );
}